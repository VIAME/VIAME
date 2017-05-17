/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
