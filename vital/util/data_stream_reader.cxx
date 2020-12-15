// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "data_stream_reader.h"

namespace kwiver {
namespace vital {

data_stream_reader::
data_stream_reader( std::istream& strm )
  : m_in_stream( strm ),
  m_line_count( 0 )
{
  m_string_editor.add( new edit_operation::shell_comment() );
  m_string_editor.add( new edit_operation::right_trim() );
  m_string_editor.add( new edit_operation::remove_blank_string() );
}

data_stream_reader::
  ~data_stream_reader()
{ }

// ------------------------------------------------------------------
bool
data_stream_reader::
getline( std::string& str )
{
  std::string line;

  while ( true )
  {
    if ( ! std::getline( m_in_stream, line ) )
    {
      // read failed.
      return false;
    }

    ++m_line_count;

    if ( m_string_editor.edit( line ) )
    {
      break;
    }
  }   // end while

  str = line;
  return true;
}

// ------------------------------------------------------------------
bool
data_stream_reader::
operator!()
{
  return ! m_in_stream.good();
}

// ------------------------------------------------------------------
size_t
data_stream_reader::
line_number() const
{
  return static_cast<size_t>(m_line_count);
}

// ------------------------------------------------------------------
void
data_stream_reader::
reset_line_number( int num )
{
  m_line_count = num;
}

// ------------------------------------------------------------------
void
data_stream_reader::
add_editor( string_edit_operation* op )
{
  m_string_editor.add( op );
}

} }   // end namespace
