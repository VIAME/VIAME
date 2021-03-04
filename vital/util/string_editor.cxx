// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "string_editor.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
string_editor::
string_editor()
{ }

string_editor::
~string_editor()
{ }

// ------------------------------------------------------------------
void
string_editor::
add( string_edit_operation* op )
{
  m_editor_list.push_back( std::shared_ptr< string_edit_operation >( op ) );
}

// ------------------------------------------------------------------
bool
string_editor::
edit( std::string& str )
{
  bool result( true );

  for( auto op : m_editor_list )
  {
    if ( ! op->process( str ) )
    {
      result = false;
      break;
    }
  }     // end foreach

  return result;
}

} }   // end namespace
