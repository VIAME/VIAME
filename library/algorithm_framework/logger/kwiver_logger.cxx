// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "kwiver_logger.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace kwiver {

namespace vital {

namespace {

// ----------------------------------------------------------------------------
/// The level a logger starts at when nothing in the environment says.
///
/// A debug build says everything and a release build says only what went
/// wrong. Imported behaviour, kept: it is what every VIAME log anyone has
/// ever looked at was produced with.
constexpr kwiver_logger::log_level_t default_level =
#if defined( NDEBUG )
  kwiver_logger::LEVEL_WARN;
#else
  kwiver_logger::LEVEL_TRACE;
#endif

// ----------------------------------------------------------------------------
/// The level named by `text`, or nothing if it names no level.
///
/// Case-insensitive, and an unrecognised name is not an error -- the logger
/// keeps the level it would have had. Rejecting it would mean a typo in a
/// shell profile stopping every VIAME process.
bool
level_from_name( std::string text, kwiver_logger::log_level_t& level )
{
  std::transform(
    text.begin(), text.end(), text.begin(),
    []( unsigned char c ){ return static_cast< char >( std::tolower( c ) ); } );

  static std::map< std::string, kwiver_logger::log_level_t > const names = {
    { "trace", kwiver_logger::LEVEL_TRACE },
    { "debug", kwiver_logger::LEVEL_DEBUG },
    { "info",  kwiver_logger::LEVEL_INFO },
    { "warn",  kwiver_logger::LEVEL_WARN },
    { "error", kwiver_logger::LEVEL_ERROR },
    { "fatal", kwiver_logger::LEVEL_FATAL },
  };

  auto const it = names.find( text );

  if( it == names.end() )
  {
    return false;
  }

  level = it->second;
  return true;
}

// ----------------------------------------------------------------------------
/// The level the environment asks for.
///
/// `VIAME_LOG_LEVEL` first, then `KWIVER_DEFAULT_LOG_LEVEL`, which is the
/// name everything written before the lite branch sets and which therefore
/// has to keep working.
kwiver_logger::log_level_t
level_from_environment()
{
  auto level = default_level;

  for( char const* const variable :
       { "VIAME_LOG_LEVEL", "KWIVER_DEFAULT_LOG_LEVEL" } )
  {
    char const* const value = std::getenv( variable );

    if( value && *value && level_from_name( value, level ) )
    {
      break;
    }
  }

  return level;
}

// ----------------------------------------------------------------------------
/// Where log lines go.
///
/// `std::cerr` always, because that is what a VIAME run's output is and what
/// reads it -- DIVE among others -- would lose if the lines went somewhere
/// else instead. `VIAME_LOG_FILE` adds a second destination rather than
/// replacing the first; it is opened once, appended to, and if it cannot be
/// opened the run says so on stderr and carries on.
class sinks
{
public:
  static sinks&
  instance()
  {
    static sinks the_sinks;
    return the_sinks;
  }

  /// Write one finished line to every sink.
  void
  write( std::string const& line )
  {
    std::lock_guard< std::mutex > guard( m_lock );

    std::cerr << line;

    if( m_file.is_open() )
    {
      m_file << line;
      m_file.flush();
    }
  }

private:
  sinks()
  {
    char const* const path = std::getenv( "VIAME_LOG_FILE" );

    if( !path || !*path )
    {
      return;
    }

    m_file.open( path, std::ios::out | std::ios::app );

    if( !m_file.is_open() )
    {
      std::cerr << "Could not open VIAME_LOG_FILE \"" << path
                << "\"; logging to standard error only\n";
    }
  }

  std::mutex m_lock;
  std::ofstream m_file;
};

// ----------------------------------------------------------------------------
/// `YYYY-MM-DD HH:MM:SS.mmm` in local time.
std::string
timestamp()
{
  using namespace std::chrono;

  auto const now = system_clock::now();
  auto const ms =
    duration_cast< milliseconds >( now.time_since_epoch() ).count() % 1000;
  auto const seconds = system_clock::to_time_t( now );

  std::tm parts{};
#if defined( _WIN32 )
  localtime_s( &parts, &seconds );
#else
  localtime_r( &seconds, &parts );
#endif

  char buffer[ 64 ];
  std::strftime( buffer, sizeof( buffer ), "%Y-%m-%d %H:%M:%S", &parts );

  std::ostringstream out;
  out << buffer << '.' << std::setfill( '0' ) << std::setw( 3 ) << ms;

  return out.str();
}

// ----------------------------------------------------------------------------
/// The loggers that exist, by name.
///
/// A logger is shared by everything that asks for its name, so that caching
/// the handle in a member is the cheap thing to do and `set_level` on it
/// means something. Locked because the first call for a given name can come
/// from any thread.
class registry
{
public:
  static registry&
  instance()
  {
    static registry the_registry;
    return the_registry;
  }

  logger_handle_t
  get( std::string const& name )
  {
    std::lock_guard< std::mutex > guard( m_lock );

    auto const it = m_loggers.find( name );

    if( it != m_loggers.end() )
    {
      return it->second;
    }

    auto handle = std::make_shared< kwiver_logger >( name );
    m_loggers[ name ] = handle;

    return handle;
  }

private:
  std::mutex m_lock;
  std::map< std::string, logger_handle_t > m_loggers;
};

} // namespace

// ----------------------------------------------------------------------------
class kwiver_logger::impl
{
public:
  explicit impl( std::string const& name )
    : m_name( name ),
      m_level( level_from_environment() )
  {}

  std::string m_name;
  log_level_t m_level;
  callback_t m_local_callback;

  static callback_t s_global_callback;
};

kwiver_logger::callback_t kwiver_logger::impl::s_global_callback = nullptr;

// ----------------------------------------------------------------------------
kwiver_logger
::kwiver_logger( std::string const& name )
  : m_impl( new kwiver_logger::impl( name ) )
{}

kwiver_logger
::~kwiver_logger() = default;

// ----------------------------------------------------------------------------
bool kwiver_logger::is_fatal_enabled() const { return m_impl->m_level <= LEVEL_FATAL; }
bool kwiver_logger::is_error_enabled() const { return m_impl->m_level <= LEVEL_ERROR; }
bool kwiver_logger::is_warn_enabled()  const { return m_impl->m_level <= LEVEL_WARN; }
bool kwiver_logger::is_info_enabled()  const { return m_impl->m_level <= LEVEL_INFO; }
bool kwiver_logger::is_debug_enabled() const { return m_impl->m_level <= LEVEL_DEBUG; }
bool kwiver_logger::is_trace_enabled() const { return m_impl->m_level <= LEVEL_TRACE; }

// ----------------------------------------------------------------------------
void
kwiver_logger
::set_level( log_level_t lev )
{
  m_impl->m_level = lev;
}

// ----------------------------------------------------------------------------
kwiver_logger::log_level_t
kwiver_logger
::get_level() const
{
  return m_impl->m_level;
}

// ----------------------------------------------------------------------------
std::string
kwiver_logger
::get_name() const
{
  return m_impl->m_name;
}

// ----------------------------------------------------------------------------
void
kwiver_logger
::set_local_callback( callback_t cb )
{
  m_impl->m_local_callback = cb;
}

// ----------------------------------------------------------------------------
void
kwiver_logger
::set_global_callback( callback_t cb )
{
  kwiver_logger::impl::s_global_callback = cb;
}

// ----------------------------------------------------------------------------
char const*
kwiver_logger
::get_level_string( kwiver_logger::log_level_t lev )
{
  switch( lev )
  {
    case kwiver_logger::LEVEL_TRACE:  return "TRACE";
    case kwiver_logger::LEVEL_DEBUG:  return "DEBUG";
    case kwiver_logger::LEVEL_INFO:   return "INFO";
    case kwiver_logger::LEVEL_WARN:   return "WARN";
    case kwiver_logger::LEVEL_ERROR:  return "ERROR";
    case kwiver_logger::LEVEL_FATAL:  return "FATAL";

    case kwiver_logger::LEVEL_NONE:
    default:           break;
  } // end switch

  return "<unknown>";
}

// ----------------------------------------------------------------------------
void
kwiver_logger
::log_message( log_level_t level, std::string const& msg )
{
  log_message( level, msg, logger_ns::location_info() );
}

// ----------------------------------------------------------------------------
void
kwiver_logger
::log_message(
  log_level_t level, std::string const& msg,
  logger_ns::location_info const& location )
{
  // The location is empty for the overload that has none, so that a message
  // logged without one is not padded out with a `?(-1): ` nobody can use.
  std::string where;

  if( location.get_line_number() >= 0 )
  {
    where = location.get_file_name() + "(" +
            std::to_string( location.get_line_number() ) + "): ";
  }

  // Every line of the message carries the whole prefix, so that a consumer
  // reading line by line never meets a line it cannot parse. Built here and
  // written once, so that two threads cannot interleave within a message.
  auto const stamp = timestamp();
  char const* const level_name = get_level_string( level );

  std::ostringstream out;
  std::istringstream in( msg );
  std::string line;

  while( std::getline( in, line ) )
  {
    out << stamp << ' ' << level_name << ' ' << where << line << '\n';
  }

  sinks::instance().write( out.str() );

  if( m_impl->m_local_callback )
  {
    m_impl->m_local_callback( level, m_impl->m_name, msg, location );
  }

  if( kwiver_logger::impl::s_global_callback )
  {
    kwiver_logger::impl::s_global_callback(
      level, m_impl->m_name, msg, location );
  }
}

// ----------------------------------------------------------------------------
// The per-level entry points. Each checks its own level, because
// `log_message` is also reachable directly.
#define VITAL_DEFINE_LOG_AT( LEVEL, level )                        \
void                                                               \
kwiver_logger                                                      \
::log_ ## level( std::string const& msg )                          \
{                                                                  \
  if( is_ ## level ## _enabled() )                                 \
  {                                                                \
    log_message( LEVEL_ ## LEVEL, msg );                           \
  }                                                                \
}                                                                  \
                                                                   \
void                                                               \
kwiver_logger                                                      \
::log_ ## level(                                                   \
  std::string const& msg, logger_ns::location_info const& location )\
{                                                                  \
  if( is_ ## level ## _enabled() )                                 \
  {                                                                \
    log_message( LEVEL_ ## LEVEL, msg, location );                 \
  }                                                                \
}

VITAL_DEFINE_LOG_AT( FATAL, fatal )
VITAL_DEFINE_LOG_AT( ERROR, error )
VITAL_DEFINE_LOG_AT( WARN,  warn )
VITAL_DEFINE_LOG_AT( INFO,  info )
VITAL_DEFINE_LOG_AT( DEBUG, debug )
VITAL_DEFINE_LOG_AT( TRACE, trace )

#undef VITAL_DEFINE_LOG_AT

// ----------------------------------------------------------------------------
logger_handle_t
get_logger( char const* name )
{
  return registry::instance().get( name );
}

// ----------------------------------------------------------------------------
logger_handle_t
get_logger( std::string const& name )
{
  return registry::instance().get( name );
}

} // namespace vital

}   // end namespace
