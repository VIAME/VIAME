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
  return m_line_count;
}


// ------------------------------------------------------------------
void
data_stream_reader::
reset_line_number( int num )
{
  m_line_count = num;
}

} }   // end namespace
