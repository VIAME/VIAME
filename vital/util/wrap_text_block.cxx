// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "wrap_text_block.h"

#include <vital/util/tokenize.h>

#include <string>
#include <list>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
wrap_text_block::
wrap_text_block()
  : m_line_length( 80 )
{
}

wrap_text_block::
~wrap_text_block()
{ }

// ------------------------------------------------------------------
void
wrap_text_block::
set_indent_string( const std::string& indent )
{
  m_indent = indent;
}

// ------------------------------------------------------------------
void
wrap_text_block::
set_line_length( size_t len )
{
  m_line_length = len;
}

// ------------------------------------------------------------------
std::string
wrap_text_block::
wrap_text( const std::string& text )
{
  std::string output_text;

  // preserve manually specified new-lines in the comment string, adding a
  // trailing new-line
  std::list< std::string > blocks;
  tokenize( text, blocks, "\n" );
  while ( blocks.size() > 0 )
  {
    std::string cur_block = blocks.front();
    blocks.pop_front();

    // always start with the comment token
    std::string line_buffer = m_indent;

    // Counter of additional spaces to place in front of the next non-empty
    // word added to the line buffer. There is always at least one space
    // between words.
    size_t spaces = 0;

    std::list< std::string > words;
    // Not using token-compress in case there is purposeful use of multiple
    // adjacent spaces, like in bullited lists. This, however, leaves open
    // the appearance of empty-string words in the loop, which are handled.
    tokenize( cur_block, words );
    while ( words.size() > 0 )
    {
      std::string cur_word = words.front();
      words.pop_front();

      // word is an empty string, meaning an intentional space was encountered.
      if ( cur_word.size() == 0 )
      {
        ++spaces;
      }
      else
      {
        if ( ( line_buffer.size() + spaces + cur_word.size() ) > m_line_length )
        {
          output_text.append( line_buffer + "\n" );
          line_buffer = m_indent;
          // On a line split, it makes sense to me that leading spaces are
          // treated as trailing white-space, which should not be output.
          spaces = 0;
        }

        line_buffer += std::string( spaces, ' ' ) + cur_word;
        spaces = 1;
      }
    } // end while words in line

    // flush remaining contents of line buffer if there is anything
    if ( line_buffer.size() > 0 )
    {
      output_text.append( line_buffer + "\n" );
    }
  } // end while lines

  return output_text;
}

} } // end namespace
