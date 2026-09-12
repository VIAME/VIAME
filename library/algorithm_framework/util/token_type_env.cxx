// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type_env.h"
#include <viame/algorithm_framework/util/file_system.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
token_type_env
::token_type_env()
  : token_type( "ENV" )
{}

// ----------------------------------------------------------------------------
token_type_env::
~token_type_env()
{}

// ----------------------------------------------------------------------------
bool
token_type_env
::lookup_entry( std::string const& name, std::string& result ) const
{
  bool retcode( true );

  const char* v = name.c_str();
  const char* env_expansion = kwiver::vital::get_env( v );
  if( env_expansion != nullptr )
  {
    result = env_expansion;
  }
  else
  {
    result.clear();
    retcode = false;
  }

  return retcode;
}

} // end namespace

} // end namespace
