// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_expand_editor.h"

#include <vital/util/token_type_env.h>
#include <vital/util/token_type_sysenv.h>

namespace kwiver {
namespace vital {

namespace edit_operation {

token_expand_editor::
token_expand_editor()
{
  // Add the default expanders
  m_token_expander.add_token_type( new kwiver::vital::token_type_env() );
  m_token_expander.add_token_type( new kwiver::vital::token_type_sysenv() );
}

token_expand_editor::
~token_expand_editor()
{ }

// ------------------------------------------------------------------
bool
token_expand_editor::
process( std::string& line )
{
  const std::string output = m_token_expander.expand_token( line );
  line = output;
  return true;
}

// ------------------------------------------------------------------
void
token_expand_editor::
add_expander( kwiver::vital::token_type * tt )
{
  m_token_expander.add_token_type( tt );
}

} } } // end namespace
