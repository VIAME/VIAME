/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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


#include "token_expander.h"

#include "token_type.h"

#include <kwiversys/RegularExpression.hxx>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** Constructor.
 *
 *
 */
token_expander::
token_expander()
  : m_logger( kwiver::vital::get_logger( "vital.token_expander" ) )
{  }

token_expander::
~token_expander()
{  }


// ----------------------------------------------------------------
/* Add token type to expander.
 *
 *
 */
bool
token_expander::
add_token_type (kwiver::vital::token_type * tt)
{
  const std::string name( tt->token_type_name() );
  m_typeList[name] = std::shared_ptr< kwiver::vital::token_type > ( tt );

  return true;
}


// ----------------------------------------------------------------
/* Look for tokens to expand.
 *
 * The syntax of the token is "$TYPE{name}".  The \c TYPE string is
 * used to locate the token type object that can provide the desired
 * text.  The \c name string, if present, is passed to the token typ
 * object to specify what result is desired.
 *
 * @param initial_string - string with token specifications embedded
 *
 * @return A string with all token references filled in.
 */
std::string
token_expander::
expand_token( std::string const& initial_string )
{
  std::string new_value;
  kwiversys::RegularExpression exp( "\\$([a-zA-Z][a-zA-Z0-9_]*)\\{([a-zA-Z0-9._:]+)?\\}" );

  std::string::const_iterator start, end;
  start = initial_string.begin();
  end = initial_string.end();

  while ( true)
  {
    std::string working_str( start, end );
    if ( ! exp.find( working_str ) )
    {
      break; // not found
    }

    // exp.match(0) - whole match
    // exp.match(1) - token type
    // exp.match(2) - optional name

    // look for the specified token type
    iterator_t ix = m_typeList.find( exp.match(1) );
    if ( ix != m_typeList.end() )
    {
      // lookup token value
      std::string result;
      if (ix->second->lookup_entry( exp.match(2), result ))
      {
        LOG_DEBUG( m_logger, "Substituting: " << "\"" << exp.match(0) << "\" -> \"" << result << "\"" );

        // append everything up to the match
        new_value.append( start, start + exp.start(0) );

        // Append the replacement string
        new_value.append( result );
      }
      else
      {
        // element type is not in the macro provider
        // append everything up to the match
        new_value.append( start, start + exp.start(0) );
        if ( handle_missing_entry( exp.match(1), exp.match(2) ) )
        {
          new_value.append( start + exp.start(0), start + exp.end(0) );
        }
      }
    }
    else
    {
      // provider type not found - no substitution, copy forward original text
      // append everything up to the match
      new_value.append( start, start + exp.start(0) );

      if ( handle_missing_provider( exp.match(1), exp.match(2) ) )
      {
        new_value.append( start + exp.start(0), start + exp.end(0) );
      }
    }

    // Update matching pointers
    start += exp.end();

  } // end while

  // copy what's left
  new_value.append( start, end );

  return new_value;
} // expand_token


// ------------------------------------------------------------------
bool
token_expander::
handle_missing_entry( const std::string& provider, const std::string& entry )
{
  // default is to insert unresolved text
  return true;
}


// ------------------------------------------------------------------
bool
token_expander::
handle_missing_provider( const std::string& provider, const std::string& entry )
{
  // default is to insert unresolved text
  return true;
}

} } // end namespace
