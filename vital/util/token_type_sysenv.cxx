// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type_sysenv.h"
#include <kwiversys/SystemTools.hxx>

#include <sstream>

#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#define HOME_ENV_NAME "UserProfile"
#define GETPID() _getpid()
#else
#include <sys/types.h>
#include <unistd.h>
#define HOME_ENV_NAME "HOME"
#define GETPID() getpid()
#endif

namespace kwiver {
namespace vital {

typedef kwiversys::SystemTools ST;

// ----------------------------------------------------------------
token_type_sysenv::
token_type_sysenv()
  : token_type ("SYSENV")
{
  m_sysinfo.RunCPUCheck();
  m_sysinfo.RunOSCheck();
  m_sysinfo.RunMemoryCheck();
}

// ----------------------------------------------------------------
token_type_sysenv::
 ~token_type_sysenv()
{ }

// ----------------------------------------------------------------
bool
token_type_sysenv::
lookup_entry (std::string const& name, std::string& result) const
{
  kwiversys::SystemInformation* SI( const_cast< kwiversys::SystemInformation* >(&m_sysinfo) );

  if ("cwd" == name) // current directory
  {
    result = ST::GetCurrentWorkingDirectory( );
    return true;
  }

  // ----------------------------------------------------------------
  if ("numproc" == name)   // number of processors/cores
  {
    std::stringstream sval;
    unsigned int numCPU = SI->GetNumberOfLogicalCPU();
    // unsigned int numCPU = SI->GetNumberOfPhysicalCPU();
    sval << numCPU;
    result = sval.str();
    return true;
  }

  // ----------------------------------------------------------------
  if ("totalvirtualmemory" == name)
  {
    std::stringstream sval;
    sval <<  SI->GetTotalVirtualMemory();
    result = sval.str();
    return true;
  }

  // ----------------------------------------------------------------
  if ("availablevirtualmemory" == name)
  {
    std::stringstream sval;
    sval << SI->GetAvailableVirtualMemory();
    result = sval.str();
    return true;
  }

  // ----------------------------------------------------------------
  if ("totalphysicalmemory" == name)
  {
    std::stringstream sval;
    sval << SI->GetTotalPhysicalMemory();
    result = sval.str();
    return true;
  }

  // ----------------------------------------------------------------
  if ("availablephysicalmemory" == name)
  {
    std::stringstream sval;
    sval << SI->GetAvailablePhysicalMemory();
    result = sval.str();
    return true;
  }

  // ----------------------------------------------------------------
  if ("hostname" == name)   // network name of system
  {
    result = SI->GetHostname();
    return true;
  }

  // ----------------------------------------------------------------
  if ("domainname" == name)
  {
    result = SI->GetFullyQualifiedDomainName();
    return true;
  }

  // ----------------------------------------------------------------
  if ("osname" == name)
  {
    result = SI->GetOSName();
    return true;
  }

  // ----------------------------------------------------------------
  if ("osdescription" == name)
  {
    result = SI->GetOSDescription();
    return true;
  }

  // ----------------------------------------------------------------
  if ("osplatform" == name)
  {
    result = SI->GetOSPlatform();
    return true;
  }

  // ----------------------------------------------------------------
  if ("osversion" == name)
  {
    result = SI->GetOSVersion();
    return true;
  }

  // ----------------------------------------------------------------
  if ("is64bits" == name)
  {
    if ( 1 == SI->Is64Bits())
    {
      result = "TRUE";
    }
    else
    {
      result = "FALSE";
    }

    return true;
  }

  // ----------------------------------------------------------------
  if ("iswindows" == name)
  {
    if ( 1 == SI->GetOSIsWindows())
    {
      result = "TRUE";
    }
    else
    {
      result = "FALSE";
    }

    return true;
  }

  // ----------------------------------------------------------------
  if ("islinux" == name)
  {
    if ( 1 == SI->GetOSIsLinux())
    {
      result = "TRUE";
    }
    else
    {
      result = "FALSE";
    }

    return true;
  }

  // ----------------------------------------------------------------
  if ("isapple" == name)
  {
    if ( 1 == SI->GetOSIsApple())
    {
      result = "TRUE";
    }
    else
    {
      result = "FALSE";
    }

    return true;
  }

  // ------------------------------------------------------------------
  if ("homedir" == name)
  {
    std::string home;
    kwiversys::SystemTools::GetEnv( HOME_ENV_NAME, home );

    if ( ! home.empty() )
    {
      result = home;
    }

    return true;
  }

  // ------------------------------------------------------------------
  if ("curdir" == name)
  {
    result = ST::GetCurrentWorkingDirectory();
    return true;
  }

  // ------------------------------------------------------------------
  if ("pid" == name)
  {
    const auto pid = GETPID();

    std::stringstream str;
    str << pid;

    result = str.str();
    return true;
  }

  return false;
}

} } // end namespace
