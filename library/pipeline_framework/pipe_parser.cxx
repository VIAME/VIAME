// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of pipeline parser.
 */

#include "pipe_parser.h"
#include "load_pipe_exception.h"

#include <viame/algorithm_framework/util/tokenize.h>
#include <viame/algorithm_framework/util/string.h>

#include <sstream>

namespace sprokit {

namespace {

#define PARSE_ERROR( T, MSG )                          \
  {                                                    \
    std::stringstream str;                             \
    str  << MSG << " at " << T->get_location();        \
    VITAL_THROW( parsing_exception, str.str() );       \
  }

#define EXPECTED( E, T ) PARSE_ERROR( T, "Expected \"" << E << "\" but found \"" << T->text() << "\"" )

#ifdef DEBUG
#define PARSER_TRACE( MSG ) LOG_TRACE( m_logger, MSG )
#else
#define PARSER_TRACE( MSG )
#endif

// ------------------------------------------------------------------
/**
 * \brief Block context structure.
 *
 * This structure holds information about the block being
 * processed. An object of this type is allocated when a "block"
 * directive is encountered. Nested blocks are managed on a stack.
 */
struct block_context_t
{
  std::string m_block_name;     // block name taken from 'block' keyword. "a:b:c"
  kwiver::vital::source_location m_location; // location where block started

  std::vector< std::string > m_previous_context;  // previous block context/name. e.g. as "a:b:c"
};

} // end namespace

// ------------------------------------------------------------------
pipe_parser
::pipe_parser()
  : m_compatibility_mode( COMPATIBILITY_ALLOW )
  , m_logger( kwiver::vital::get_logger( "sprokit.pipe_parser" ) )
{
}

// ------------------------------------------------------------------
void
pipe_parser
::add_search_path( kwiver::vital::config_path_t const& file_path )
{
  m_lexer.add_search_path( file_path );
}

// ------------------------------------------------------------------
void
pipe_parser
::add_search_path( kwiver::vital::config_path_list_t const& file_path )
{
  m_lexer.add_search_path( file_path );
}

// ------------------------------------------------------------------
void
pipe_parser
::set_compatibility_mode( compatibility_mode_t mode )
{
  m_compatibility_mode = mode;
}

// ------------------------------------------------------------------
/**
 * \brief Parse a process pipeline file
 *
 * Grammar:
 * process-pipe-file ::= <def-list>
 *
 * def-list ::= <pipe-def-item>
 *            | <pipe-def-item> <def-list>
 *
 * pipe-def-item ::= <config-block>
 *                 | <config-block>
 *                 | <process-definition>
 *                 | <process-connection>
 *
 */
sprokit::pipe_blocks
pipe_parser
::parse_pipeline( std::istream& input, const std::string& name )
{
  m_lexer.open_stream( input, name );

  while( true )
  {
    auto t = m_lexer.get_token();
    PARSER_TRACE( "Got " << *t );

    if ( t->token_type() == TK_CONFIG )
    {
      m_lexer.unget_token( t );
      config_pipe_block cpb;
      process_config_block( cpb );
      m_pipe_blocks.push_back( cpb );
      continue;
    }

    if ( t->token_type() == TK_PROCESS )
    {
      m_lexer.unget_token( t );
      process_pipe_block ppb;
      process_definition(ppb);
      m_pipe_blocks.push_back( ppb );
      continue;
    }

    if ( t->token_type() == TK_CONNECT )
    {
      m_lexer.unget_token( t );
      connect_pipe_block cpb;
      process_connection( cpb );
      m_pipe_blocks.push_back( cpb );
      continue;
    }

    if ( t->token_type() >= TK_EOL )
    {
      break;
    }

    PARSE_ERROR( t, "Found unexpected token \"" << t->text() << "\"");
  } // end while

  return m_pipe_blocks;
}

// ==================================================================
/**
 * \brief Handle process definition production
 *
 * "process" <proc_name> "::" <proc-type> <opt-config>
 */
void
pipe_parser
::process_definition(process_pipe_block& ppb)
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that his is a "process" token
  expect_token( TK_PROCESS, t );

  // Save location of the "process" keyword
  ppb.loc = t->get_location();

  t = m_lexer.get_token();
  if (t->token_type() != TK_IDENTIFIER)
  {
    EXPECTED( "process name", t );
  }

  ppb.name = t->text();

  t = m_lexer.get_token();      // get '::' separator
  if (t->token_type() != TK_DOUBLE_COLON)
  {
    EXPECTED( "\"::\"", t );
  }

  t = m_lexer.get_token();      // get from process type
  if (t->token_type() != TK_IDENTIFIER)
  {
    EXPECTED( "process type name", t );
  }

  ppb.type = t->text();
  PARSER_TRACE( "Accepted process definition - " << ppb.name << " :: " << ppb.type );

  // Handle the optional config lines
  parse_config( ppb.config_values );
}

// ------------------------------------------------------------------
/**
 * \brief Parse top level config block.
 *
 * config-block ::= "config" <config-key><EOL> <config-entry-list>
 */
void
pipe_parser
::process_config_block( config_pipe_block& cpb )
{
  auto t = m_lexer.get_token();
  PARSER_TRACE( "process_config_block Got " << *t );

  // Should be guaranteed that this is a config token
  expect_token( TK_CONFIG, t );

  // Save location of the "config" keyword
  cpb.loc = t->get_location();

  // initiate reporting of EOL
  m_lexer.absorb_eol( false );

  // identifier token is requried
  t = m_lexer.get_token();
  PARSER_TRACE( "process_config_block block name Got " << *t );

  // expecting TK_IDENTIFIER ":" TK_IDENTIFIER... <eol>
  if ( t->token_type() == TK_IDENTIFIER )
  {
    cpb.key.push_back( t->text() );
  }
  else
  {
    EXPECTED( "config block key component", t );
  }

  // Process the rest of the block name. Need to split up name into components.
  // Loop terminates at EOL
  while ( true )
  {
     t = m_lexer.get_token();
     PARSER_TRACE( "process_config_block block name-2 Got " << *t );

     if ( t->token_type() >= TK_EOL)
     {
       break;
     }

     // This must be a ':' <IDENTIFIER> if we have not hit EOL
     if ( t->token_type() != TK_COLON)
     {
       EXPECTED( "\":\"", t );
     }

     // expecting TK_IDENTIFIER which always follows a ':'
     t = m_lexer.get_token();
     if ( t->token_type() == TK_IDENTIFIER )
     {
       cpb.key.push_back( t->text() );
     }
     else
     {
       EXPECTED( "config block key component", t );
     }
  } // end while

  PARSER_TRACE( "Accept config block header - " << kwiver::vital::join(cpb.key, ":") );

  m_lexer.absorb_eol( true );

  // process config lines in block
  parse_config( cpb.values );
}

// ------------------------------------------------------------------
/**
 * \brief Parse old style config entries.
 *
 * ":"<key><opt-attrs><whitespace><value>
 *
 * key ::= id : id : ...
 * key ::= static/id : id : ...
 *
 * opt_attrs ::=
 *             | "[" <attr-list> "]"
 *
 * attr-list ::= attr
 *             | attr <attr_list>
 */
void
pipe_parser
::old_config( sprokit::config_value_t& val )
{
  // Note that the leading ':' has been absorbed

  // We need whitepsace tokens to parse the old format.
  m_lexer.absorb_whitespace( false );
  token_sptr t;

  // process the block name. Need to split up name into components
  while ( true )
  {
    std::string part = parse_config_key();
    val.key_path.push_back( part );

    t = m_lexer.get_token();
    if ( t->token_type() != TK_COLON)
    {
      break;
    }
  } // end while

  // expecting '[' or something else (a space)
  if ( t->token_value() == '[' )
  {
    // process attrs
    parse_attrs( val );
    t = m_lexer.get_token();
  }

  // should be a space
  if ( t->token_value() != TK_WHITESPACE )
  {
    EXPECTED( "whitespace", t );
  }

  // save rest of line as the config value
  val.loc = t->get_location();
  val.value = m_lexer.get_rest_of_line();

  m_lexer.absorb_whitespace( true );

  PARSER_TRACE( "Accepted old style config: \"" << kwiver::vital::join( val.key_path, ":" ) << "\""
                << " = " << "\"" << val.value << "\"" );
}

// ------------------------------------------------------------------
/**
 * \brief Parse new style config entries.
 *
 * <opt-relativepath><key><opt-attrs> = <value>
 *
 * key ::= id : id ...
 *
 * opt_attrs ::=
 *             | "[" <attr-list> "]"
 *
 * attr-list ::= attr
 *             | attr <attr_list>
 *
 */
void
pipe_parser
::new_config( sprokit::config_value_t& val )
{
  token_sptr t;

  // process the entry name. Need to split up name into components
  while ( true )
  {
    std::string part = parse_config_key();
    val.key_path.push_back( part );

    // look for key component separator
    t = m_lexer.get_token();
    if ( t->token_type() != TK_COLON)
    {
      break;
    }
  } // end while

  // expecting '[' or '=' or ':='
  if ( t->token_value() == '[' )
  {
    // process attrs
    parse_attrs( val );

    // get next token after ']'
    t = m_lexer.get_token();
  }

  if ( t->token_value() == TK_ASSIGN
    || t->token_value() == TK_LOCAL_ASSIGN )
  {
    if ( t->token_value() == TK_LOCAL_ASSIGN )
    {
      val.flags.push_back( "local-assign" );
    }

    // save rest of line as the config value
    val.value = t->text();
    val.loc = t->get_location();
  }
  else
  {
    PARSE_ERROR( t, "Expecting assignment operator but found \"" << t->text() << "\"" );
  }

    PARSER_TRACE( "Accepted new style config: \"" << kwiver::vital::join( val.key_path, ":" ) << "\""
                << " = " << "\"" << val.value << "\"" );
}

// ------------------------------------------------------------------
/**
 * \brief Parse attribute list.
 *
 * This method parses the attribute list. The leading '[' has already
 * been absorbed. All attributes up to the closing ']' are added to
 * the attribute list. The closing ']' is also absorbed.
 *
 * Note that the attributes are rather free-form at this point. We
 * could verify each one at this point, but historically, that is done
 * at a later step (semantic validation).
 *
 * '[' <attr-list> ']'
 *
 * attr-list ::= attr
 *             | attr ',' attr_list
 *
 * \param[out] val Attributes are set in this parameter
 */
void
pipe_parser
::parse_attrs( sprokit::config_value_t& val )
{
  auto t = m_lexer.get_token();

  while( true )
  {
    // The original grammar does not force flag names to be reserved
    // words. Since flag names can appear in other contexts where they
    // are not to be treated as flags, we have to accept an IDENTIFIER
    // here rather than use an ATTRIBUTE token type.
    if ( t->token_type() != TK_IDENTIFIER )
    {
      PARSE_ERROR( t, "Expecting attribute flag but found \"" << t->text() << "\"" );
    }

    val.flags.push_back( t->text() );

    t = m_lexer.get_token();

    if ( t->token_value() == ']' )
    {
      return;
    }

    if ( t->token_value() != ',' )
    {
      PARSE_ERROR( t, "Expecting ',' but found \"" << t->text() << "\"" );
    }

    // The token *after* the comma is the next flag. Without this the comma
    // itself went round the loop and was rejected for not being a flag
    // name, so `attr ',' attr_list` -- which the grammar above has always
    // claimed -- could not be parsed at all.
    t = m_lexer.get_token();
  } // end while
}

// ------------------------------------------------------------------
/**
 * \brief Connection production
 *
 * process-connection ::= "connect" "from" <proc>"."<port> "to" <proc>"."<port>
 */
void
pipe_parser
::process_connection( connect_pipe_block& cpb )
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that this is a connect token
  expect_token( TK_CONNECT, t );

  // Save location of the "connect" keyword
  cpb.loc = t->get_location();

  // Validate noise word "from"
  t = m_lexer.get_token();
  expect_token( TK_FROM, t );

  parse_port_addr( cpb.from );

  // Validate noise word "to"
  t = m_lexer.get_token();
  expect_token( TK_TO, t );

  parse_port_addr( cpb.to );
  PARSER_TRACE( "Accept connect" );
}

// ------------------------------------------------------------------
/**
 * \brief Parse port addr specification.
 *
 * port_addr ::= <proc_name> '.' <port_name>
 * port_name ::= <id>
 *             | <id> '\' <port_name>
 *
 * \param[out] out_pa The port address parts are returned here.
 */
void
pipe_parser
::parse_port_addr( process::port_addr_t& out_pa)
{
  const std::string proc_name = parse_process_name();

  auto t = m_lexer.get_token();      // get separator
  expect_token( '.', t );

  std::string port_name = parse_port_name();

  // copy output to parameter
  out_pa = process::port_addr_t( proc_name, port_name );
}

// ------------------------------------------------------------------
/**
 * \brief Parse extended id.
 *
 * This method parses an extended identifier. It is just like a
 * regular ID token but also allows some special characters.
 *
 * key-comp ::= <id>
 *            |  <id> <extra-char> <key-comp>
 *
 * \param extra_char Additional separator characters that are
 * allowable in this class of identifier.
 *
 * \param expecting Semantic name of identifier being parsed. Used for
 * identifying errors
 *
 * \return Text of key component
 */
std::string
pipe_parser
::parse_extended_id( const std::string& extra_char,
                   const std::string& expecting)
{
  std::string comp;

  token_sptr t;

  while (true )
  {
    t = m_lexer.get_token();
    if (t->token_type() != TK_IDENTIFIER )
    {
      EXPECTED( expecting, t ); // terminate with error
    }

    comp += t->text();

    // handle allowable characters
    t = m_lexer.get_token();
    auto pos = extra_char.find( t->token_value() );
    if ( pos != std::string::npos ) // char allowable
    {
      comp += extra_char[pos];
      continue;
    }

    // unallowable character found. Terminate identifier.
    break;
  }

  // put unknown token back
  m_lexer.unget_token( t );

  return comp;
}

// ------------------------------------------------------------------
std::string
pipe_parser
::parse_config_key()
{
  return parse_extended_id( "/.", "config key component" );
}

// ------------------------------------------------------------------
std::string
pipe_parser
::parse_port_name()
{
  return parse_extended_id( "/", "port name" );
}

// ------------------------------------------------------------------
std::string
pipe_parser
::parse_process_name()
{
  return parse_extended_id( "/", "process name" );
}

// ------------------------------------------------------------------
/**
 * \brief Parse series of config lines.
 *
 * This method parses zero or more sequential config
 * specifications. Both new and old style config entries are handled.
 *
 * \param[out] out_config Vector updated with config entries parsed.
 */
void
pipe_parser
::parse_config( config_values_t& out_config )
{
  // current config context
  std::vector< std::string > current_context;

  // nested block stack
  std::vector< block_context_t > block_stack;

  while ( true )
  {
    config_value_t config_val;

    // look at current token
    auto t = m_lexer.get_token();
    PARSER_TRACE( "parse_config first get  " << *t );

    if ( t->token_type() == TK_BLOCK )
    {
      // handle block name

      const std::string block_name = m_lexer.get_rest_of_line();

      // Save current block context and start another
      block_context_t block_ctxt;
      block_ctxt.m_block_name = block_name; // block name
      block_ctxt.m_location = m_lexer.current_location();
      block_ctxt.m_previous_context = current_context;

      kwiver::vital::tokenize( block_name, current_context, ":", kwiver::vital::TokenizeTrimEmpty );

      block_stack.push_back( block_ctxt );
      continue;
    }

    if ( t->token_type() == TK_ENDBLOCK )
    {
      // handle endblock keyword

      m_lexer.flush_line();

      if ( block_stack.empty() )
      {
        PARSE_ERROR( t, "\"endblock\" found without matching \"block\"" );
      }

      // Restore previous block context
      current_context = block_stack.back().m_previous_context;
      block_stack.pop_back( );
      continue;
    }

    // preload keypath with current block context
    config_val.key_path = current_context;
    m_lexer.unget_token( t );
    if ( ! parse_config_line( config_val ) )
    {
      // not a valid config line. Could be something else valid though.
      break;
    }

    // Add config entry to the current list.
    out_config.push_back( config_val );

  } // ---- end while ----

  // ------ end of config section ------
  // validate that the block stack is empty
  if ( 0 != block_stack.size() )
  {
    std::stringstream msg;

    msg << "Unclosed blocks left at end of config section:\n";
    while ( 0 != block_stack.size() )
    {
      msg << "Block \"" << block_stack.back().m_block_name
          << "\" - Started at " << block_stack.back().m_location
          << std::endl;

      block_stack.pop_back();
    }

    VITAL_THROW( parsing_exception, msg.str() );
  }
}

// ------------------------------------------------------------------
/**
 * \brief Process a single config line.
 *
 * This method processes a single config line of either the old or new
 * style. If the config line is valid, the output parameter is updated.
 *
 * \param[out] config_val Updated with config information.
 *
 * \return \b true if valid config line processed. This means that the
 * return parameter is valid. \b false indicates that the line was not
 * a valid config line and the output parameter is not valid.
 */
bool
pipe_parser
::parse_config_line( config_value_t& config_val )
{
  bool ret_status(true);        // assume o.k.
  auto t = m_lexer.get_token();
  PARSER_TRACE( "parse_config_line Got " << *t );

  // leading colon indicates old format config entry
  if ( t->token_type() == TK_COLON )
  {
    if ( COMPATIBILITY_WARN == m_compatibility_mode )
    {
      LOG_WARN( m_logger, "Old style config specification found at " << t->get_location() );
    }
    else if ( COMPATIBILITY_ERROR == m_compatibility_mode )
    {
      LOG_ERROR( m_logger, "Old style config specification found at " << t->get_location() );
      ret_status = false;
    }

    old_config( config_val );

    return ret_status;
  }

  // Check for relative path specifier. This is a good indication
  // that this is a new style config entry
  if ( t->token_type() == TK_RELATIVE_PATH )
  {
    config_val.flags.push_back( "relativepath" );

    // get next token
    t = m_lexer.get_token();
  }

  // possibly a new style config
  // <key> "=" <value>
  // <key>.<key> "=" <value>
  // <key>:<key> "=" <value>
  // <key>"["<attr-list"]" "=" <value>
  // Best test is presence of '=', but that is hard to do.
  if ( t->token_type() == TK_IDENTIFIER )
  {
    // push last two tokens back to lexer
    m_lexer.unget_token( t );

    // process new style config
    new_config( config_val );

    return ret_status;
  }

  // probably not an config statement
  m_lexer.unget_token( t );

  // indicate some other type of token
  return false;
}

// ------------------------------------------------------------------
bool
pipe_parser
::expect_token( int expected_tk, token_sptr t )
{
  if ( t->token_value() != expected_tk )
  {
    PARSE_ERROR( t, "Expected \"" << token::token_name( expected_tk ) << "\" keyword, but found "
               << t->text() );
  }

  return true;
}

} // end namespace
