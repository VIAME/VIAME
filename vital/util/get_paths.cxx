// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
    int length(0);
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
    char ppath[4096];
    int length(0);
    wai_getModulePath( ppath, sizeof ppath, &length );
    path[length] = '\0';

    // convert to string
    path = ppath;
  }

  return path;
}

} }
