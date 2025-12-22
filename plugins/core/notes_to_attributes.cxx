/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

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
  else if( str[0] == '(' )
  {
    return delim + str;
  }
  else if( str.find_first_of( ',' ) != std::string::npos )
  {
    return delim + "(note) \"" + str + "\"";
  }
  else
  {
    return delim + "(note) " + str;
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

        output += delim + "(atr) " + category + " " + value;

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

    if( attr.empty() || attr[0] != '(' )
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

    if( col[0] == "(kp)" )
    {
      if( col.size() != 4 )
      {
        continue; // throw error
      }

      detection.add_keypoint( col[1], { std::stod( col[2] ),
                                        std::stod( col[3] ) } );
    }
    else if( col[0] == "(note)" )
    {
      std::string full_note = attr.substr( 7 );

      if( attr.size() > 7 && attr[7] == '\"' && attr.back() != '\"' )
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
    else if( col[0] == "(atr)" )
    {
      if( col.size() < 2 )
      {
        continue; // throw error
      }

      std::string formatted_note = ":" + col[1];

      if( col.size() == 2 )
      {
        formatted_note += "=true";
      }
      else if( col.size() > 2 )
      {
        formatted_note += "=" + col[2];

        for( unsigned i = 3; i < col.size(); ++i )
        {
          formatted_note += " " + col[i];
        }
      }

      detection.add_note( formatted_note );
    }
  }
}

} // end namespace
