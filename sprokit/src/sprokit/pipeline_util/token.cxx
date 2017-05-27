/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include "token.h"

namespace sprokit {

// ------------------------------------------------------------------
token::token()
  : m_token_type( TK_EOF )
{
}


// ------------------------------------------------------------------
token::token( int c )
{
  if ( c < TK_FIRST )
  {
    m_token_type = TK_NONE;
    m_text = std::string( 1, (char) c );
  }
  else
  {
    m_token_type = c;
    m_text = std::string( 1, (char) c );
  }
}


// ------------------------------------------------------------------
token::token( int type, const std::string& s )
  : m_token_type( type )
  , m_text( s )
{ }


// ------------------------------------------------------------------
token::~token()
{ }


// ------------------------------------------------------------------
int
token::token_value() const
{
  if ( m_token_type == TK_NONE )
  {
    return ( m_text.c_str() )[0];
  }

  return m_token_type;
}


// ------------------------------------------------------------------
const char*
token::token_name( int tk )
{
#define C( T, D ) case T:  return D
  static char msg[128];  // kluge

  if ( tk < TK_FIRST )
  {
    sprintf( msg, "Character '%c'", tk );
    return msg;
  }

  switch ( tk )
  {
    C( TK_EOF, "end-of-file" );
    C( TK_IDENTIFIER, "identifier" );
    C( TK_CLUSTER_DESC, "cluster_descrip" );
    C( TK_LOCAL_ASSIGN, "local-assignment" );
    C( TK_ASSIGN, "assignment" );
    C( TK_COLON, "start-of-legacy-config" );
    C( TK_PROCESS, "process" );
    C( TK_CONNECT, "connect" );
    C( TK_FROM, "from" );
    C( TK_TO, "to" );
    C( TK_RELATIVE_PATH, "relativepath" );
    C( TK_STATIC, "static" );
    C( TK_BLOCK, "block" );
    C( TK_ENDBLOCK, "endblock" );
    C( TK_CONFIG, "config" );
    C( TK_CLUSTER, "cluster" );
    C( TK_IMAP, "imap" );
    C( TK_OMAP, "omap" );
    C( TK_ATTRIBUTE, "ATTRIBUTE" );
    C( TK_DOUBLE_COLON, "DOUBLE_COLON" );
    C( TK_WHITESPACE, "WHITESPACE" );
    C( TK_EOL, "EOL" );

  default:
    return "Unknown token";
  } // end switch

#undef C
}


// ------------------------------------------------------------------
std::ostream&
token::format( std::ostream& str ) const
{
  str << "token: ";
#define C( T ) case T:  str << # T; break

  switch ( m_token_type )
  {
    C( TK_EOF );
    C( TK_IDENTIFIER );
    C( TK_CLUSTER_DESC );
    C( TK_LOCAL_ASSIGN );
    C( TK_ASSIGN );
    C( TK_COLON );
    C( TK_PROCESS );
    C( TK_CONNECT );
    C( TK_FROM );
    C( TK_TO );
    C( TK_RELATIVE_PATH );
    C( TK_STATIC );
    C( TK_BLOCK );
    C( TK_ENDBLOCK );
    C( TK_CLUSTER );
    C( TK_CONFIG );
    C( TK_IMAP );
    C( TK_OMAP );
    C( TK_ATTRIBUTE );
    C( TK_DOUBLE_COLON );
    C( TK_WHITESPACE );
    C( TK_EOL );

  case TK_NONE:
    str << "Character token: ";
    break;

  default:
    str << "***Unknown token type: " << m_token_type;
    break;
  } // end switch

#undef C

  str << "  '" << m_text << "'\n"
      << "    At: " << m_srcLocation << std::endl;

  return str;
}

} // end namespace
