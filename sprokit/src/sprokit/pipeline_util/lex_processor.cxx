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

#include "lex_processor.h"

#include <vital/exceptions.h>
#include <vital/config/config_block_exception.h>
#include <vital/util/data_stream_reader.h>
#include <kwiversys/SystemTools.hxx>

#include <fstream>
#include <istream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <set>


/*
Design notes.

Hopefully lots of stuff can be borrowed from the config parser. If
there are shared methods, there will need to be a common class with
public methods that support those operations.

May need to borrow the block context class too since we *MUST* support
the block/endblock keywords. The block stack really belongs in the parser
not in the lexer.

Debugging output should include tracing input stream and logging token
output stream.

 */

namespace sprokit {

namespace {

// ----------------------------------------------------------------
/**
 * @brief Context for file processing.
 *
 * This object represents a file being processed. An object of this
 * type can be pushed on a stack when an include directive is found to
 * keep track of where to resume when the included file is processed.
 */
class include_context
{
public:
  include_context( const std::string& file_name )
    : m_stream( file_name ) // open file stream
    , m_reader( m_stream ) // assign stream to reader
    , m_filename( std::make_shared< std::string >(file_name) )
  {
    if ( ! m_stream )
    {
      throw kwiver::vital::config_file_not_found_exception( file_name, "could not open file");
    }
  }


  include_context( const include_context& other )
    : m_stream( *other.m_filename )
    , m_reader( m_stream )
    , m_filename( other.m_filename )
  {
  }


  ~include_context()
  {
    m_stream.close();
  }

  /**
   * @brief Get name of file.
   */
  const std::string& file() const { return *m_filename; }

  // -- MEMBER DATA --

  // This is the actual file stream
  std::ifstream m_stream;

  // This reader operates on the above stream to provide trimmed input
  // with no comments or blank lines
  kwiver::vital::data_stream_reader m_reader;
  std::shared_ptr< std::string > m_filename;
};

} // end namespace


// ==================================================================
class lex_processor::priv
{
public:
  priv();

  /**
   * @brief Find reserved word.
   *
   * The supplied string is checked against the list of reserved
   * words. If it is a reserved word, then the associated token code
   * is returned. If it is not a reserved word, the non-specific WORD
   * token value is returned.
   *
   * @param n String to determine if it is a reserved word.
   *
   * @return The token code for the string is returned.
   */
  int find_res_word( const std::string& n ) const;

  token_sptr process_id();

  /**
   * @brief Get current source location.
   *
   * This method returns the file and line number of the current
   * source file. This location also tracks all nested include files.
   *
   * @return The current source location (file and line) is returned.
   */
  kwiver::vital::source_location current_loc() const;

  /**
   * @brief Get new input line
   *
   *
   * @return \b true if line is returned. \b false if EOF
   */
  bool get_line();

  /**
   * @brief Trim spaces from string/
   *
   * This method removes leading and trailing spaces from the supplied
   * string. The string is returned in place.
   *
   * @param[in,out] str String to be trimmed.
   *
   * @return \b true is returned if the string has been changed.
   */
  bool trim_string( std::string& str );

  /**
   * @brief Flush current line.
   *
   */
  void flush_line();

  kwiver::vital::config_path_t resolve_file_name( kwiver::vital::config_path_t const& file_name );

  //------------------------------------------------------------------
  // This indicates the first token in the line. It is set when a new
  // line is read in and reset after the first token is extracted from
  // that line.
  bool m_first_token_in_line;

  // -- the following data do not need to be initialized by CTOR

  /** Current input line that is being processed.
   */
  std::string m_input_line;
  std::string::iterator m_cur_char;

  /** Stack for push back of tokens. The lexer client can push back a
   * token or two if the parser needs to give up on a particular path.
   */
  std::vector< token_sptr > m_token_stack;

  /// Translates keyword to token type
  std::map< std::string, int > m_keyword_table;

  /**
   * This field is the include file stack. The top entry on the stack
   * is the current file. An entry is pushed on the stack when an
   * include directive is encountered. An entry is popped off the
   * stack at end of file and the previous file is resumed, unless it
   * is the last entry on the stack when a real EOF token is retrned.
   */
  std::vector< include_context > m_include_stack;

  /**
   * This is used to remove leading and trailing strings.
   */
  kwiver::vital::string_editor m_trim_string;

  // file search path list
  kwiver::vital::config_path_list_t m_search_path;
};


/* ---------------------------------------------------
 * Constructor
 */
lex_processor::
lex_processor()
  : m_priv( new lex_processor::priv )
{ }


lex_processor::
~lex_processor()
{ }


/* ----------------------------------------------------
 * This method runs as a thread and processes all
 * of the specified files.
 */
void
lex_processor::
open_file( const std::string& file_name )
{
  // open file, throw error if open error
  m_priv->m_include_stack.push_back( include_context( file_name ) );
}


// ------------------------------------------------------------------
kwiver::vital::source_location
lex_processor::
current_location() const
{
  return m_priv->current_loc();
}


// ------------------------------------------------------------------
std::string
lex_processor::
get_rest_of_line()
{
  std::string line( m_priv->m_cur_char, m_priv->m_input_line.end() );
  m_priv->trim_string( line );

  m_priv->flush_line();
  return line;
}


// ------------------------------------------------------------------
void
lex_processor::
add_search_path( kwiver::vital::config_path_t const& file_path )
{
  m_priv->m_search_path.push_back( file_path );
}


// ------------------------------------------------------------------
void
lex_processor::
add_search_path( kwiver::vital::config_path_list_t const& file_path )
{
  m_priv->m_search_path.insert( m_priv->m_search_path.end(),
                              file_path.begin(), file_path.end() );
}


// ------------------------------------------------------------------
void
lex_processor::
unget_token( token_sptr token )
{
  m_priv->m_token_stack.push_back( token );
}


// ---------------------------------------------------
token_sptr
lex_processor::
get_token()
{
  // First check the token stack
  if ( ! m_priv->m_token_stack.empty() )
  {
    auto t = m_priv->m_token_stack.back();
    m_priv->m_token_stack.pop_back();
    return t;
  }

  if ( m_priv->m_cur_char == m_priv->m_input_line.end() )
  {
    // get new line
    while ( ! m_priv->get_line() )
    {
      // EOF encountered
      m_priv->m_include_stack.pop_back();

      // check for the real end of input
      if ( m_priv->m_include_stack.empty() )
      {
        // return EOF token
        return std::make_shared< token > ( TK_EOF, "E-O-F" );
      }
    }

    // get new line could check for "include" keyword and handle that operation.
    // Check for include directive - starts with "include "
    if ( m_priv->m_input_line.substr( 0, 8 ) != "include " )
    {
      // process include directive
      m_priv->m_cur_char += 8;

      std::string file_name( m_priv->m_cur_char, m_priv->m_input_line.end() );
      m_priv->trim_string( file_name );

      kwiver::vital::config_path_t resolv_filename = m_priv->resolve_file_name( file_name );
      if ( "" == resolv_filename ) // could not resolve
      {
        std::ostringstream sstr;
        sstr << "file included from " << current_location()
             << " could not be found in search path.";

        throw kwiver::vital::config_file_not_found_exception( file_name, sstr.str() );
      }

      // Push the current location onto the include stack
      m_priv->m_include_stack.push_back( include_context( resolv_filename ) );
    }
  }

  // loop until we have discovered a token
  while ( 1 )
  {
    token_sptr t;
    char c = *m_priv->m_cur_char++;  // Advance pointer

    if ( std::isspace( c ) )
    {
      continue;
    }

    if ( '=' == c )
    {
      // assignment operator
      t = std::make_shared< token > ( '=' );
      t->set_location( m_priv->current_loc() );
      return t;
    }

    if ( m_priv->m_first_token_in_line )
    {
      if ( ':' == c )  // old style config line
      {
        m_priv->m_first_token_in_line = false;
        t = std::make_shared< token > ( ':' );
        t->set_location( m_priv->current_loc() );
        return t;
      }
    }

    // Look for cluster documentation. "--"
    if ( ( '-' == c ) && ( '-' == *m_priv->m_cur_char ) )
    {
      std::string text( m_priv->m_cur_char + 1, m_priv->m_input_line.end() );
      m_priv->trim_string( text );
      t = std::make_shared< token > ( TK_CLUSTER_DESC, text );
      t->set_location( m_priv->current_loc() );

      m_priv->flush_line();
      return t;
    }

    // Is it a character.
    // Then start of a word
    if ( isalnum( c ) || ( c == '_' ) )
    {
      --m_priv->m_cur_char;
      t = m_priv->process_id();
      return t;
    }

    // All that's left is single char tokens return c as character

    // At this point,just pass all single characters
    // as tokens.  Let the parser decide what to do.
    m_priv->m_first_token_in_line = false;
    t = std::make_shared< token > ( c );
    t->set_location( m_priv->current_loc() );
    return t;
  }   // end while
} // lex_processor::get_next_token


// ==================================================================
lex_processor::priv::
priv()
  : m_first_token_in_line( false )
{
  m_cur_char = m_input_line.end();
  m_trim_string.add( new kwiver::vital::edit_operation::left_trim );
  m_trim_string.add( new kwiver::vital::edit_operation::right_trim );

  //
  // Fill in the keyword map. An alternate approach would be to use
  // gperf.
  //
  m_keyword_table["block"]        = TK_BLOCK;
  m_keyword_table["endblock"]     = TK_ENDBLOCK;
  m_keyword_table["process"]      = TK_PROCESS;
  m_keyword_table["/static"]      = TK_STATIC;
  m_keyword_table["ro"]           = TK_RO;

  // These append keywords could be shorteded
  m_keyword_table["append"]       = TK_APPEND;
  m_keyword_table["append-sp"]    = TK_APPEND_SP;
  m_keyword_table["append-comma"] = TK_APPEND_COMMA;
  m_keyword_table["append-path"]  = TK_APPEND_PATH;

  m_keyword_table["connect"]      = TK_CONNECT;
  m_keyword_table["from"]         = TK_FROM;
  m_keyword_table["to"]           = TK_TO;

  m_keyword_table["cluster"]      = TK_CLUSTER;
  m_keyword_table["imap"]         = TK_IMAP;
  m_keyword_table["omap"]         = TK_OMAP;
  m_keyword_table["relativepath"] = TK_RELATIVE_PATH;
}


// ------------------------------------------------------------------
token_sptr
lex_processor::priv::
process_id()
{
  int a = *m_cur_char++;

  std::string ident( 1, (char) a ); // Start the ident with a character

  while ( m_cur_char != m_input_line.end() )
  {
    a = *m_cur_char++; // Advance pointer

    // These characters are only allowed in the first token
    // (e.g. config key)
    if ( ( ( a == '.' ) || ( a == ':' ) ) &&
         ! m_first_token_in_line )
    {
      break;
    }

    // Is the character one of [a-zA-Z0-9_.:]
    if ( isalnum( a ) || ( a == '_' ) || ( a == '.' ) || ( a == ':' ) )
    {
      ident += a; // add the character
    }
    else
    {
      // back up iterator
      --m_cur_char;
      break;
    }
  }   // end while

  // Check ident against list of keywords
  // Create the new token
  token_sptr t = std::make_shared< token > ( find_res_word( ident ), ident );
  t->set_location( current_loc() );
  m_first_token_in_line = false;

  return t;
}


//------------------------------------------------------------------
/*
 * This function returns the token that is associated with
 * the suppliied name. If the name is not in the table,
 * TK_IDENTIFIER token is returned.
 */
int
lex_processor::priv::
find_res_word( const std::string& n ) const
{
  if ( m_keyword_table.count( n )  > 0 )
  {
    return m_keyword_table.at(n);
  }

  return TK_IDENTIFIER;     // not in table
}


// ------------------------------------------------------------------
kwiver::vital::source_location
lex_processor::priv::
current_loc() const
{
  // Get current location from the include file stack top element
  return kwiver::vital::source_location( m_include_stack.front().m_filename,
                                         m_include_stack.front().m_reader.line_number() );
}


// ------------------------------------------------------------------
bool
lex_processor::priv::
get_line()
{
  bool status = m_include_stack.back().m_reader.getline( m_input_line );

  if ( status )
  {
    m_cur_char = m_input_line.begin();
    m_first_token_in_line = true;
  }
  return status;
}


// ------------------------------------------------------------------
bool
lex_processor::priv::
trim_string( std::string& str )
{
  return m_trim_string.edit( str );
}


// ------------------------------------------------------------------
/**
 * @brief Flush remaining line in parser.
 *
 * This method causes a new line to be read from the file.
 */
void
lex_processor::priv::
flush_line()
{
  m_input_line.clear();
  m_cur_char = m_input_line.end();
}


// ------------------------------------------------------------------
/**
 * @brief Resolve file name against search path.
 *
 * This method returns a valid file path, including name, for the
 * supplied file_name using the currently active file search path.
 *
 * If the file can not be found in the current search path, then the
 * nested include directories are searched in nested order.
 *
 * A null string is returned if the file can not be found anywhere.
 *
 * @param file File name to resolve.
 *
 * @return Full file path, or empty string on failure.
 */
kwiver::vital::config_path_t
lex_processor::priv::
resolve_file_name( kwiver::vital::config_path_t const& file_name )
{
  // Test for absolute file name
  if ( kwiversys::SystemTools::FileIsFullPath( file_name ) )
  {
    return file_name;
  }

  // The file is on a relative path.
  // See if file can be found in the search path.
  std::string res_file =
    kwiversys::SystemTools::FindFile( file_name, this->m_search_path, true );
  if ( "" != res_file )
  {
    return res_file;
  }

  // File not found in regular path, search backwards in current
  // include stack. First we have to reverse the include stack and
  // remove duplicate paths.
  std::set< std::string > dir_set;
  kwiver::vital::config_path_list_t include_paths;
  const auto eit = m_include_stack.rend();
  for ( auto it = m_include_stack.rbegin(); it != eit; ++it )
  {
    kwiver::vital::config_path_t config_file_dir( kwiversys::SystemTools::GetFilenamePath( it->file() ) );
    if ( "" == config_file_dir )
    {
      config_file_dir = ".";
    }

    if ( 0 == dir_set.count( config_file_dir ) )
    {
      dir_set.insert( config_file_dir );
      include_paths.push_back( config_file_dir );
    }
  }     // end for

  return kwiversys::SystemTools::FindFile( file_name, include_paths, true );
}

} // end namespace
