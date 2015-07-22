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

#include <boost/lexical_cast.hpp>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
token_type_sysenv::
token_type_sysenv()
  : token_type ("SYSENV")
{ }


// ----------------------------------------------------------------
token_type_sysenv::
 ~token_type_sysenv()
{ }


// ----------------------------------------------------------------
bool
token_type_sysenv::
lookup_entry (std::string const& name, std::string& result)
{

  if ("cwd" == name) // current directory
  {
    // This just-in-time caching approach may give unexpected results
    // if the program changes CWD dynamically.
    static std::string s_cwd;

    if ( s_cwd.empty() )
    {
      // current directory
      char path[FILENAME_MAX];

      if ( getcwd( path, sizeof(path) ) )
      {
        path[sizeof(path) - 1] = '\0'; // force end of string
        s_cwd = path;
      }
    }

    result = s_cwd;
    return true;
  }

  // ----------------------------------------------------------------
  else if ("numproc" == name)   // number of processors/cores
  {
    unsigned int numCPU = m_sysinfo.GetNumberOfPhysicalCPU();
    result = boost::lexical_cast< std::string > ( numCPU );
    return true;
  }

  // ----------------------------------------------------------------
  else if ("hostname" == name)   // network name of system
  {
    result = m_sysinfo.GetHostname();
    return true;
  }

  // ----------------------------------------------------------------
  else if ("domainname" == name)
  {
    result = m_sysinfo.GetFullyQualifiedDomainName();
    return true;
  }

  // ----------------------------------------------------------------
  else if ("osname" == name)
  {
    result = m_sysinfo.GetOSName();\
    return true;
  }

  /*
  const char * GetOSRelease();
  const char * GetOSVersion();
  const char * GetOSPlatform();
  */
  // ----------------------------------------------------------------
//+  else if ("date" == name)
//  {

//  }

  return false;
}

} // end namespace
} // end namespace
