// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type_sysenv.h"
#include <viame/algorithm_framework/util/file_system.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>

#if defined( _WIN32 ) || defined( _WIN64 )
#include <process.h>
#include <windows.h>
#include <winsock2.h>
#define HOME_ENV_NAME "UserProfile"
#define GETPID() _getpid()
#else
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>
#define HOME_ENV_NAME "HOME"
#define GETPID() getpid()
#if defined( __APPLE__ )
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#endif

namespace viame {

namespace {

// ----------------------------------------------------------------------------
/// What the four memory names answer, in megabytes.
///
/// Megabytes, and **"virtual" means swap**. That is not what the word
/// suggests, but it is what these names have always answered -- measured,
/// not assumed: on this machine `totalvirtualmemory` was `SwapTotal` and
/// `availablevirtualmemory` was `SwapFree`, to the megabyte. Renaming them
/// would be a change to the configuration language, which this is not.
struct memory_sizes
{
  size_t total_physical = 0;
  size_t available_physical = 0;
  size_t total_virtual = 0;
  size_t available_virtual = 0;
};

#if defined( __linux__ )
// ----------------------------------------------------------------------------
/// One `/proc/meminfo` figure, in kilobytes, or zero.
size_t
meminfo_value( std::string const& text, std::string const& key )
{
  auto const at = text.find( "\n" + key + ":" );
  auto const from = ( at == std::string::npos && text.rfind( key + ":", 0 ) == 0 )
                    ? key.size() + 1
                    : ( at == std::string::npos
                        ? std::string::npos
                        : at + key.size() + 2 );

  if( from == std::string::npos )
  {
    return 0;
  }

  return static_cast< size_t >( std::strtoull( text.c_str() + from, nullptr, 10 ) );
}
#endif

// ----------------------------------------------------------------------------
memory_sizes
system_memory()
{
  memory_sizes sizes;

#if defined( _WIN32 ) || defined( _WIN64 )
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof( status );

  if( GlobalMemoryStatusEx( &status ) )
  {
    constexpr unsigned long long mb = 1024ull * 1024ull;
    sizes.total_physical = static_cast< size_t >( status.ullTotalPhys / mb );
    sizes.available_physical =
      static_cast< size_t >( status.ullAvailPhys / mb );
    sizes.total_virtual = static_cast< size_t >( status.ullTotalPageFile / mb );
    sizes.available_virtual =
      static_cast< size_t >( status.ullAvailPageFile / mb );
  }
#elif defined( __linux__ )
  // `/proc/meminfo` rather than `sysinfo()`, because `sysinfo` has no figure
  // for the page cache and the answer for available memory has always
  // counted it. Kilobytes there, megabytes here.
  std::ifstream meminfo( "/proc/meminfo" );

  if( meminfo.is_open() )
  {
    std::string const text{ std::istreambuf_iterator< char >( meminfo ),
                            std::istreambuf_iterator< char >() };

    auto const mb = []( size_t kb ){ return kb / 1024; };

    sizes.total_physical = mb( meminfo_value( text, "MemTotal" ) );
    sizes.available_physical = mb( meminfo_value( text, "MemFree" ) +
                                   meminfo_value( text, "Buffers" ) +
                                   meminfo_value( text, "Cached" ) );
    sizes.total_virtual = mb( meminfo_value( text, "SwapTotal" ) );
    sizes.available_virtual = mb( meminfo_value( text, "SwapFree" ) );
  }
#elif defined( __APPLE__ )
  int64_t physical = 0;
  size_t length = sizeof( physical );

  if( sysctlbyname( "hw.memsize", &physical, &length, nullptr, 0 ) == 0 )
  {
    sizes.total_physical =
      static_cast< size_t >( physical / ( 1024 * 1024 ) );
  }

  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

  if( host_statistics64(
        mach_host_self(), HOST_VM_INFO64,
        reinterpret_cast< host_info64_t >( &vm ), &count ) == KERN_SUCCESS )
  {
    auto const page = static_cast< uint64_t >( sysconf( _SC_PAGESIZE ) );
    auto const free_bytes =
      ( static_cast< uint64_t >( vm.free_count ) +
        static_cast< uint64_t >( vm.inactive_count ) ) * page;

    sizes.available_physical =
      static_cast< size_t >( free_bytes / ( 1024 * 1024 ) );
  }

  // Swap, which is what the two "virtual" names mean here.
  struct xsw_usage swap{};
  length = sizeof( swap );

  if( sysctlbyname( "vm.swapusage", &swap, &length, nullptr, 0 ) == 0 )
  {
    sizes.total_virtual =
      static_cast< size_t >( swap.xsu_total / ( 1024 * 1024 ) );
    sizes.available_virtual =
      static_cast< size_t >( swap.xsu_avail / ( 1024 * 1024 ) );
  }
#endif

  return sizes;
}

// ----------------------------------------------------------------------------
/// The short host name, as `hostname` prints it.
std::string
host_name()
{
  char buffer[ 256 ] = { 0 };

  if( ::gethostname( buffer, sizeof( buffer ) - 1 ) != 0 )
  {
    return {};
  }

  return buffer;
}

// ----------------------------------------------------------------------------
/// The fully qualified name, or the short one if nothing resolves it.
///
/// **Through the network interfaces, not the host name.** Resolving the host
/// name finds whatever `/etc/hosts` says, which on a normal machine is the
/// short name against 127.0.1.1; the qualified name is only known to the
/// reverse lookup of an address the machine actually answers on. Measured:
/// on this host, resolving `kana` gives `kana`, and reversing its interface
/// address gives `kana.kitware.com`, which is what this has always answered.
std::string
domain_name()
{
  auto const host = host_name();

#if !defined( _WIN32 ) && !defined( _WIN64 )
  struct ifaddrs* interfaces = nullptr;

  if( ::getifaddrs( &interfaces ) == 0 )
  {
    std::string qualified;

    for( auto* entry = interfaces; qualified.empty() && entry;
         entry = entry->ifa_next )
    {
      if( !entry->ifa_addr || ( entry->ifa_flags & IFF_LOOPBACK ) )
      {
        continue;
      }

      socklen_t length = 0;

      if( entry->ifa_addr->sa_family == AF_INET )
      {
        length = sizeof( struct sockaddr_in );
      }
      else if( entry->ifa_addr->sa_family == AF_INET6 )
      {
        length = sizeof( struct sockaddr_in6 );
      }
      else
      {
        continue;
      }

      char name[ NI_MAXHOST ] = { 0 };

      // The first *qualified* one. A docker bridge reverses to the short
      // name, and taking that would be worse than taking nothing.
      if( ::getnameinfo(
            entry->ifa_addr, length, name, sizeof( name ),
            nullptr, 0, NI_NAMEREQD ) == 0 &&
          std::strchr( name, '.' ) )
      {
        qualified = name;
      }
    }

    ::freeifaddrs( interfaces );

    if( !qualified.empty() )
    {
      return qualified;
    }
  }
#endif

  return host;
}

// ----------------------------------------------------------------------------
/// `uname`'s three strings: what the system calls itself, what release it
/// is, and the build it was made from.
struct os_strings
{
  std::string name;
  std::string release;
  std::string version;
  std::string machine;
};

// ----------------------------------------------------------------------------
os_strings
operating_system()
{
  os_strings strings;

#if defined( _WIN32 ) || defined( _WIN64 )
  strings.name = "Windows";

  SYSTEM_INFO info{};
  GetNativeSystemInfo( &info );

  switch( info.wProcessorArchitecture )
  {
    case PROCESSOR_ARCHITECTURE_AMD64: strings.machine = "x86_64"; break;
    case PROCESSOR_ARCHITECTURE_ARM64: strings.machine = "arm64";  break;
    case PROCESSOR_ARCHITECTURE_INTEL: strings.machine = "x86";    break;
    default:                           strings.machine = "unknown"; break;
  }
#else
  struct utsname info{};

  if( ::uname( &info ) == 0 )
  {
    strings.name = info.sysname;
    strings.release = info.release;
    strings.version = info.version;
    strings.machine = info.machine;
  }
#endif

  return strings;
}

} // namespace

// ----------------------------------------------------------------------------
token_type_sysenv
::token_type_sysenv()
  : token_type( "SYSENV" )
{}

// ----------------------------------------------------------------------------
token_type_sysenv::
~token_type_sysenv()
{}

// ----------------------------------------------------------------------------
bool
token_type_sysenv
::lookup_entry( std::string const& name, std::string& result ) const
{
  auto const number =
    []( size_t value ){
      std::stringstream out;
      out << value;
      return out.str();
    };

  auto const boolean =
    []( bool value ){ return value ? std::string( "TRUE" )
                                   : std::string( "FALSE" ); };

  // --------------------------------------------------------------------------
  if( "cwd" == name || "curdir" == name )
  {
    result = viame::current_working_directory();
    return true;
  }

  // --------------------------------------------------------------------------
  if( "numproc" == name )   // number of processors/cores
  {
    // The logical count, which is what kwiversys reported and what a caller
    // sizing a thread pool wants.
    auto const count = std::thread::hardware_concurrency();
    result = number( count ? count : 1u );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "totalvirtualmemory" == name )
  {
    result = number( system_memory().total_virtual );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "availablevirtualmemory" == name )
  {
    result = number( system_memory().available_virtual );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "totalphysicalmemory" == name )
  {
    result = number( system_memory().total_physical );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "availablephysicalmemory" == name )
  {
    result = number( system_memory().available_physical );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "hostname" == name )   // network name of system
  {
    result = host_name();
    return true;
  }

  // --------------------------------------------------------------------------
  if( "domainname" == name )
  {
    result = domain_name();
    return true;
  }

  // --------------------------------------------------------------------------
  if( "osname" == name )
  {
    result = operating_system().name;
    return true;
  }

  // --------------------------------------------------------------------------
  if( "osdescription" == name )
  {
    auto const os = operating_system();
    result = os.name + " " + os.release + " " + os.version;
    return true;
  }

  // --------------------------------------------------------------------------
  if( "osplatform" == name )
  {
    result = operating_system().machine;
    return true;
  }

  // --------------------------------------------------------------------------
  if( "osversion" == name )
  {
    // `uname`'s *version*, not its release: the build string, which is what
    // this has always answered.
    result = operating_system().version;
    return true;
  }

  // --------------------------------------------------------------------------
  if( "is64bits" == name )
  {
    result = boolean( sizeof( void* ) == 8 );
    return true;
  }

  // --------------------------------------------------------------------------
  if( "iswindows" == name )
  {
#if defined( _WIN32 ) || defined( _WIN64 )
    result = boolean( true );
#else
    result = boolean( false );
#endif
    return true;
  }

  // --------------------------------------------------------------------------
  if( "islinux" == name )
  {
#if defined( __linux__ )
    result = boolean( true );
#else
    result = boolean( false );
#endif
    return true;
  }

  // --------------------------------------------------------------------------
  if( "isapple" == name )
  {
#if defined( __APPLE__ )
    result = boolean( true );
#else
    result = boolean( false );
#endif
    return true;
  }

  // --------------------------------------------------------------------------
  if( "homedir" == name )
  {
    std::string home;
    viame::get_env( HOME_ENV_NAME, home );

    if( !home.empty() )
    {
      result = home;
    }

    return true;
  }

  // --------------------------------------------------------------------------
  if( "pid" == name )
  {
    std::stringstream str;
    str << GETPID();

    result = str.str();
    return true;
  }

  return false;
}

} // namespace viame
