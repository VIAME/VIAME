// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "demangle.h"

#include <viame/algorithm_framework/vital_config.h>

#if VITAL_USE_ABI_DEMANGLE

#include <cstdlib>
#include <cxxabi.h>

#endif

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
std::string
demangle( std::string const& sym )
{
  return demangle( sym.c_str() );
}

// ----------------------------------------------------------------------------
std::string
demangle( char const* sym )
{
#if VITAL_USE_ABI_DEMANGLE
  std::string tname( sym );
  int status;
  char* demangled_name = abi::__cxa_demangle( sym, nullptr, nullptr, &status );

  if( 0 == status )
  {
    tname = demangled_name;
    std::free( demangled_name );
  }

  return tname;
#else
  return sym;
#endif
}

} // namespace vital

}   // end namespace
