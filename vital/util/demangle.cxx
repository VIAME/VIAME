// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "demangle.h"

#include <vital/vital_config.h>

#if VITAL_USE_ABI_DEMANGLE

#include <cxxabi.h>
#include <cstdlib>

#endif

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
std::string demangle( std::string const& sym )
{
  return demangle( sym.c_str() );
}

// ------------------------------------------------------------------
std::string demangle( char const* sym )
{

#if VITAL_USE_ABI_DEMANGLE

  std::string tname( sym );
  int status;
  char* demangled_name = abi::__cxa_demangle(sym, NULL, NULL, &status);

  if( 0 == status )
  {
    tname = demangled_name;
    std::free(demangled_name);
  }

  return tname;

#else

  return sym;

#endif
}

} } // end namespace
