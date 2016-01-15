/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

/**
 * \file
 * \brief Wrapper over C functions to get executable path and module path.
 */

#include <vital/util/get_paths.h>

namespace kwiver {
namespace vital{

// Code originally from https://github.com/gpakosz/whereami.git
// and used unmodified

// make support functions static
#define WAI_FUNCSPEC static
#include <vital/util/whereami.h>
#include "whereami.c"


// ------------------------------------------------------------------
std::string
get_executable_path()
{
  static std::string path; // cached path

  if ( path.empty() )
  {
    char ppath[4096];
    int length;
    wai_getExecutablePath( ppath, sizeof ppath, &length );
    ppath[length] = '\0';

    // convert to string
    path = ppath;
  }

  return path;
}


// ------------------------------------------------------------------
std::string
get_module_path()
{
  static std::string path; // cached path

  if ( path.empty() )
  {
    int length = wai_getModulePath( NULL, 0, NULL );
    char ppath[4096];
    wai_getModulePath( ppath, sizeof ppath, &length );
    path[length] = '\0';

    // convert to string
    path = ppath;
  }

  return path;
}

} }
