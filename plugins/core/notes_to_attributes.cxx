/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "notes_to_attributes.h"

#include <vital/util/tokenize.h>

namespace viame
{

std::string
format_note( const std::string& str, const std::string delim )
{
  if( str.empty() )
  {
    return "";
  }
  else if( str[0] == '+' )
  {
    return delim + str;
  }
  else if( str.find_first_of( ',' ) != std::string::npos )
  {
    return delim + "+note \"" + str + "\"";
  }
  else
  {
    return delim + "+note " + str;
  }
}

std::string
notes_to_attributes( const std::vector< std::string >& notes,
                     const std::string delim )
{
  std::string output;

  for( std::string note : notes )
  {
    while( !note.empty() )
    {
      std::size_t pos = note.find_first_of( ':' );
      std::size_t pos2 = note.find_first_of( '=' );

      if( pos == std::string::npos || pos2 == std::string::npos || pos2 == pos + 1 )
      {
        output += format_note( note, delim );
        break;
      }
      else if( note[0] == ' ' )
      {
        note = note.substr( 1 );
      }
      else if( pos != 0 )
      {
        output += format_note( note.substr( 0, pos ), delim );
        note = note.substr( pos );
      }
      else if( note[0] == ':' )
      {
        note = note.substr( 1 );
        pos2--;

        pos = note.find_first_of( ':' );

        while( pos != std::string::npos && pos > 0 && note[pos-1] == ' ' )
        {
          pos--;
        }

        std::size_t value_len = ( pos == std::string::npos ?
                                  note.size() - pos2 - 1 :
                                  pos - pos2 - 1 );

        std::string category = note.substr( 0, pos2 );
        std::string value = note.substr( pos2 + 1, value_len );

        output += delim + "+atr " + category + " " + value;

        if( pos == std::string::npos )
        {
          break;
        }
        else
        {
          note = note.substr( pos );
        }
      }
    }
  }

  return output;
}

void
add_attributes_to_detection( kwiver::vital::detected_object& detection,
                             const std::vector< std::string >& attrs )
{
  for( unsigned i = 0; i < attrs.size(); ++i )
  {
    const std::string& attr = attrs[i];

    if( attr.empty() || attr[0] != '+' )
    {
      continue;
    }

    // tokensize attribute
    std::vector< std::string > col;
    kwiver::vital::tokenize( attr, col, " ", false );

    if( col.empty() )
    {
      continue;
    }

    if( col[0] == "+kp" )
    {
      if( col.size() != 4 )
      {
        continue; // throw error
      }

      detection.add_keypoint( col[1], { std::stod( col[2] ),
                                        std::stod( col[3] ) } );
    }
    else if( col[0] == "+note" )
    {
      std::string full_note = attr.substr( 6 );

      if( attr.size() > 6 && attr[6] == '\"' && attr.back() != '\"' )
      {
        for( unsigned j = i + 1; j < attrs.size(); j++ )
        {
          full_note += "," + attrs[j];

          i++;

          if( attrs[j].back() == '\"' )
          {
            break;
          }
        }
      }

      detection.add_note( full_note );
    }
    else if( col[0] == "+atr" )
    {
      if( col.size() < 3 )
      {
        continue; // throw error
      }

      std::string formatted_note = ":" + col[1] + "=" + col[2];

      for( unsigned i = 3; i < col.size(); ++i )
      {
        formatted_note += " " + col[i];
      }

      detection.add_note( formatted_note );
    }
  }
}

} // end namespace
