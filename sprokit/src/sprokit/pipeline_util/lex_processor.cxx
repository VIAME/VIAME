// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "lex_processor.h"

#include <vital/exceptions.h>
#include <vital/config/config_block_exception.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/string.h>
#include <vital/util/token_expander.h>
#include <vital/util/token_type_sysenv.h>
#include <vital/util/token_type_env.h>
#include <kwiversys/SystemTools.hxx>

#include <sprokit/pipeline_util/load_pipe_exception.h>

#include <fstream>
#include <istream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <set>

// Make value evaluate to true to enable low level lexer debugging of
// token traffic
#define LEX_DEBUG 0

using kst = kwiversys::SystemTools;

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
    : m_fstream( file_name, std::ios_base::in ) // open file stream
    , m_stream( &m_fstream )
    , m_reader( m_fstream ) // assign stream to reader
    , m_filename( std::make_shared< std::string >( kst::GetRealPath(file_name) ) )
  {
    if ( ! m_stream )
    {
      VITAL_THROW( kwiver::vital::config_file_not_found_exception, file_name, "could not open file");
    }

    // make sure file is readable
    if ( ! kst::TestFileAccess( file_name, kwiversys::TEST_FILE_OK | kwiversys::TEST_FILE_READ ) )
    {
      VITAL_THROW( kwiver::vital::config_file_not_found_exception, file_name, "could not access file");
    }
  }

  include_context( std::istream& str, const std::string& file_name )
    : m_stream( &str ) // open file stream
    , m_reader( *m_stream ) // assign stream to reader
    , m_filename( std::make_shared< std::string >( kwiversys::SystemTools::GetRealPath(file_name) ) )
  {
  }

  ~include_context() = default;

  /**
   * @brief Get name of file.
   */
  const std::string& file() const { return *m_filename; }

  // -- MEMBER DATA --

  // This is the actual file stream
  std::fstream m_fstream;
  std::istream* m_stream;

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
   * @brief Flush current line.
   *
   */
  void flush_line();

  kwiver::vital::config_path_t resolve_file_name( kwiver::vital::config_path_t const& file_name );

  //------------------------------------------------------------------
  // These absorb* attributes are not the prettiest way of handling
  // conditional separators, but the original grammar used whitespace
  // and EOL as tokens. Generally the new grammar does not care about
  // these but there are some cases where it is needed to support the
  // old grammar. Rather than adding many optional whitespace and EOL
  // tokens in all the productions, those that need it turn on these
  // rarely used tokens.

  // This indicates whether EOL's should be absorbed or reported.
  bool m_absorb_eol;

  // This indicates whether whitespace should be absorbed or reported.
  bool m_absorb_whitespace;

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
  std::vector< std::shared_ptr< include_context > > m_include_stack;

  // file search path list
  kwiver::vital::config_path_list_t m_search_path;

  kwiver::vital::token_expander m_token_expander;
};

/* ---------------------------------------------------
 * Constructor
 */
lex_processor
::lex_processor()
  : m_logger( kwiver::vital::get_logger( "sprokit.pipe_processor" ) )
  , m_priv( new lex_processor::priv )
{ }

lex_processor
::~lex_processor()
{ }

// ------------------------------------------------------------------
void
lex_processor
::open_file( const std::string& file_name )
{
  // open file, throw error if open error
  m_priv->m_include_stack.push_back( std::make_shared< include_context >( file_name ) );
}

// ------------------------------------------------------------------
void
lex_processor
::open_stream( std::istream& input, const std::string& file_name )
{
  m_priv->m_include_stack.push_back( std::make_shared< include_context >( input, file_name ) );
}

// ------------------------------------------------------------------
kwiver::vital::source_location
lex_processor
::current_location() const
{
  return m_priv->current_loc();
}

// ------------------------------------------------------------------
std::string
lex_processor
::get_rest_of_line()
{
  std::string line( m_priv->m_cur_char, m_priv->m_input_line.end() );
  kwiver::vital::string_trim( line );

  m_priv->flush_line();
  return line;
}

// ------------------------------------------------------------------
void
lex_processor
::flush_line()
{
  m_priv->flush_line();
}

// ------------------------------------------------------------------
void
lex_processor
::add_search_path( kwiver::vital::config_path_t const& file_path )
{
  m_priv->m_search_path.push_back( file_path );
  LOG_DEBUG( m_logger, "Adding \"" << file_path << "\" to search path" );
}

// ------------------------------------------------------------------
void
lex_processor
::add_search_path( kwiver::vital::config_path_list_t const& file_path )
{
  m_priv->m_search_path.insert( m_priv->m_search_path.end(),
                              file_path.begin(), file_path.end() );

  LOG_DEBUG( m_logger, "Adding \"" << kwiver::vital::join( file_path, ", " )
             << "\" to search path" );
}

// ------------------------------------------------------------------
void
lex_processor
::unget_token( token_sptr token )
{
  m_priv->m_token_stack.push_back( token );

#if LEX_DEBUG
  LOG_TRACE( m_logger, "Ungetting " << *token );
#endif
}

// ---------------------------------------------------
token_sptr
lex_processor
::get_token()
{
  auto t = get_next_token();

#if LEX_DEBUG
  LOG_TRACE( m_logger, *t );
#endif

  return t;
}

// ---------------------------------------------------
token_sptr
lex_processor
::get_next_token()
{
  // First check the token stack
  if ( ! m_priv->m_token_stack.empty() )
  {
    auto t = m_priv->m_token_stack.back();
    m_priv->m_token_stack.pop_back();
    return t;
  }

  // check for the real end of input
  if ( m_priv->m_include_stack.empty() )
  {
    // return EOF token
    return std::make_shared< token > ( TK_EOF, "E-O-F" );
  }

  if ( m_priv->m_cur_char == m_priv->m_input_line.end() )
  {
    // get new line
    if ( ! get_next_line() )
    {
      // return EOF token
      return std::make_shared< token > ( TK_EOF, "E-O-F" );
    }

    if ( ! m_priv->m_absorb_eol )
    {
      auto t = std::make_shared< token > ( TK_EOL, "" );
      t->set_location( current_location() );
      return t;
    }
  } // end of EOL handling

  // loop until we have discovered a token
  while ( m_priv->m_cur_char != m_priv->m_input_line.end() )
  {
    token_sptr t;
    char c = *m_priv->m_cur_char++;  // Advance pointer

    if ( std::isspace( c ) )
    {
      if ( m_priv->m_absorb_whitespace )
      {
        continue;
      }
      else
      {
        // generate a whitespace token collecting all whitespace at
        // this point
        t = std::make_shared< token > ( TK_WHITESPACE, " " );
        t->set_location( current_location() );

        // Skip over all whitespace. No need to generate more tokens.
        while ( ( m_priv->m_cur_char != m_priv->m_input_line.end() )
                && ( std::isspace( *m_priv->m_cur_char ) ) )
        {
          ++m_priv->m_cur_char;
        }

        return t;
      }
    }

    // Check to see if there is another character in look-ahead memory
    if ( m_priv->m_cur_char != m_priv->m_input_line.end() )
    {
      char n = *m_priv->m_cur_char++;  // Advance pointer

      // Look for cluster documentation. "--"
      if ( ( '-' == c ) && ( '-' == n ) )
      {
        std::string text;
        if ( m_priv->m_cur_char != m_priv->m_input_line.end() )
        {
          // Collect description text from after token to EOL
          text = std::string{ m_priv->m_cur_char + 1, m_priv->m_input_line.end() };
          kwiver::vital::string_trim( text );
        }
        else
        {
          text = "\n"; // blank line causes line break
        }

        t = std::make_shared< token > ( TK_CLUSTER_DESC, text );
        t->set_location( current_location() );

        m_priv->flush_line();
        return t;
      }

      // look for "::"
      if ( c == ':' && n == ':' )
      {
        token_sptr tok = std::make_shared< token > ( m_priv->find_res_word( "::" ), "::" );
        tok->set_location( current_location() );

        return tok;
      }

      if ( c == ':' && n == '=' )
      {
        // Collect rest of line as text
        std::string text( m_priv->m_cur_char + 1, m_priv->m_input_line.end() );
        kwiver::vital::string_trim( text );
        token_sptr tok = std::make_shared< token > ( m_priv->find_res_word( ":=" ), text );
        tok->set_location( current_location() );

        m_priv->flush_line();
        return tok;
      }

      // Reset pointer to restore our second character
      m_priv->m_cur_char--;
    }

    if ( '=' == c )
    {
      // assignment operator
      // Collect rest of line as text
      std::string text( m_priv->m_cur_char, m_priv->m_input_line.end() );
      kwiver::vital::string_trim( text );
      t = std::make_shared< token > ( TK_ASSIGN, text );
      t->set_location( current_location() );

      m_priv->flush_line();
      return t;
    }

    if ( ':' == c )  // old style config line
    {
      t = std::make_shared< token > ( TK_COLON, ":" );
      t->set_location( current_location() );
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
    t = std::make_shared< token > ( c );
    t->set_location( current_location() );
    return t;
  }   // end while

  // This probably should not happen since blank lines are absorbed.
  // Although, including empty files will get us here.
  return get_next_token();
} // lex_processor::get_next_token

// ----------------------------------------------------------------------------
bool
lex_processor
::get_next_line()
{
  // get new line
  while ( !m_priv->get_line() )
  {
    // EOF encountered
    LOG_TRACE( m_logger, "End of file on \""
               << m_priv->m_include_stack.back()->file()
               << "\"" );
    m_priv->m_include_stack.pop_back();

    // check for the real end of input
    if ( m_priv->m_include_stack.empty() )
    {
      // return EOF
      return false;
    }
  }   // end while

  // get new line could check for "include" keyword and handle that operation.
  // Check for include directive - starts with "include " or "!include"
  if ( *m_priv->m_cur_char == '!' )
  {
    m_priv->m_cur_char++;
  }

  if ( kwiver::vital::starts_with( std::string( m_priv->m_cur_char,
                                                   m_priv->m_input_line.end() ),
                                      "include " ) )
  {
    // process include directive
    m_priv->m_cur_char += 8;

    std::string file_name( m_priv->m_cur_char, m_priv->m_input_line.end() );
    kwiver::vital::string_trim( file_name );

    // Perform macro substitutions first
    file_name = m_priv->m_token_expander.expand_token( file_name );

    kwiver::vital::config_path_t resolv_filename = m_priv->resolve_file_name(
      file_name );
    if ( "" == resolv_filename )   // could not resolve
    {
      std::ostringstream sstr;
      sstr << file_name << " included from " << current_location() <<
        " could not be found in search path.";

      VITAL_THROW( sprokit::file_no_exist_exception, sstr.str() );
    }

    LOG_TRACE( m_logger, "Including file: \"" << resolv_filename << "\"" );
    m_priv->flush_line();

    // Push the current location onto the include stack
    m_priv->m_include_stack.push_back( std::make_shared< include_context >(
                                         resolv_filename ) );

    // Get first line from included file.
    return get_next_line();
  } // end include

  return true;
}

// ------------------------------------------------------------------
void
lex_processor
::absorb_eol( bool opt )
{
  m_priv->m_absorb_eol = opt;
}

// ------------------------------------------------------------------
void
lex_processor
::absorb_whitespace( bool opt )
{
  m_priv->m_absorb_whitespace = opt;
}

// ==================================================================
lex_processor::priv
::priv()
  : m_absorb_eol( true )
  , m_absorb_whitespace( true )
{
  m_cur_char = m_input_line.end();

  //
  // Fill in the keyword map. An alternate approach would be to use
  // gperf.
  //
  m_keyword_table["block"]        = TK_BLOCK;
  m_keyword_table["endblock"]     = TK_ENDBLOCK;
  m_keyword_table["process"]      = TK_PROCESS;

  m_keyword_table["connect"]      = TK_CONNECT;
  m_keyword_table["from"]         = TK_FROM;
  m_keyword_table["to"]           = TK_TO;

  m_keyword_table["cluster"]      = TK_CLUSTER;
  m_keyword_table["imap"]         = TK_IMAP;
  m_keyword_table["omap"]         = TK_OMAP;
  m_keyword_table["relativepath"] = TK_RELATIVE_PATH;
  m_keyword_table["config"]       = TK_CONFIG;

  m_keyword_table["::"]           = TK_DOUBLE_COLON;
  m_keyword_table[":="]           = TK_LOCAL_ASSIGN;

  m_token_expander.add_token_type( new kwiver::vital::token_type_env() );
  m_token_expander.add_token_type( new kwiver::vital::token_type_sysenv() );
}

// ------------------------------------------------------------------
token_sptr
lex_processor::priv
::process_id()
{
  int a;

  std::string ident;

  while ( m_cur_char != m_input_line.end() )
  {
    a = *m_cur_char++; // Advance pointer

    // Is the character one of [a-zA-Z0-9_-]
    if ( isalnum( a )
         || ( a == '-' )
         || ( a == '_' )
      )
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

  return t;
}

//------------------------------------------------------------------
/*
 * This function returns the token that is associated with
 * the suppliied name. If the name is not in the table,
 * TK_IDENTIFIER token is returned.
 */
int
lex_processor::priv
::find_res_word( const std::string& n ) const
{
  if ( m_keyword_table.count( n )  > 0 )
  {
    return m_keyword_table.at(n);
  }

  return TK_IDENTIFIER;     // not in table
}

// ------------------------------------------------------------------
kwiver::vital::source_location
lex_processor::priv
::current_loc() const
{
  // Get current location from the include file stack top element
  return kwiver::vital::source_location( m_include_stack.back()->m_filename,
           static_cast<int>(m_include_stack.back()->m_reader.line_number()) );
}

// ------------------------------------------------------------------
bool
lex_processor::priv
::get_line()
{
  bool status = m_include_stack.back()->m_reader.getline( m_input_line );

  if ( status )
  {
    kwiver::vital::string_trim( m_input_line );
    m_cur_char = m_input_line.begin();
  }
  return status;
}

// ------------------------------------------------------------------
/**
 * @brief Flush remaining line in parser.
 *
 * This method causes a new line to be read from the file. It is
 * idempotent in that multiple calls will not flush multiple lines.
 */
void
lex_processor::priv
::flush_line()
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
lex_processor::priv
::resolve_file_name( kwiver::vital::config_path_t const& file_name )
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
    kwiver::vital::config_path_t config_file_dir( kwiversys::SystemTools::GetFilenamePath( (*it)->file() ) );
    if ( "" == config_file_dir )
    {
      config_file_dir = ".";
    }

    if ( 0 == dir_set.count( config_file_dir ) )
    {
      dir_set.insert( config_file_dir );
      include_paths.push_back( config_file_dir );
    }
  }

  res_file = kwiversys::SystemTools::FindFile( file_name, include_paths, true );

  if ( "" != res_file )
  {
    return res_file;
  }

  // Lastly, as a last resort, see if file can be found in a local directory.
  std::vector< std::string > relative_path( 1, "." );
  return kwiversys::SystemTools::FindFile( file_name, relative_path, true );
}

} // end namespace
