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

/**
 * \file
 * \brief Implementation of pipeline parser.
 */

#include "pipe_parser.h"
#include "pipe_parser_exception.h"

#include <vital/util/tokenize.h>

#include <sstream>

/*
 * Open issues:
 *
 * Supporting the old version of the "providers" is problematic and
 * leads to ugly syntax of the config entries. Currently not supported.
 *
 * Support for ":=" local assignment. This is useful in the
 * stand-alone config parser, but it is not as compelling in the pipe
 * parser since we can use pipe level config blocks for the same
 * purpose. Given that we must be able to ingest config files that are
 * acceptable to the config_parser, we must handle this in some
 * manner. One possible approach is to annotate this operator in the
 * config attributes and build a local symbol table during an early
 * pass over the AST. This local table would be available for fill-ins
 * when the config block is created.
 */

namespace sprokit {

namespace {

#define PARSE_ERROR( T, MSG )                   \
  {                                             \
    std::stringstream str;                      \
    str  << MSG << " at " << T->get_location(); \
    throw parsing_exception( str.str() );       \
  }

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
pipe_parser::
pipe_parser()
  : m_compatibility_mode( COMPATIBILITY_ALLOW )
  , m_logger( kwiver::vital::get_logger( "sprokit.pipe_parser" ) )
{
}


// ------------------------------------------------------------------
void
pipe_parser::
add_search_path( kwiver::vital::config_path_t const& file_path )
{
  m_lexer.add_search_path( file_path );
}


void
pipe_parser::
add_search_path( kwiver::vital::config_path_list_t const& file_path )
{
  m_lexer.add_search_path( file_path );
}


// ------------------------------------------------------------------
void
pipe_parser::
set_compatibility_mode( compatibility_mode_t mode )
{
  m_compatibility_mode = mode;
}


// ------------------------------------------------------------------
sprokit::pipe_blocks
pipe_parser::
parse_pipeline( std::istream& input )
{
  while( true )
  {
    auto t = m_lexer.get_token();

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

    PARSE_ERROR( t, "Found unexpected token \"" << t->text() << "\"");
  } // end while

  return m_pipe_blocks;
}


// ------------------------------------------------------------------
/**
 * @brief Process a cluster pipe block
 *
 * cluster <name><eol>
 *        -- description
 *       <cpb> <cluster-proc>
 *
 * cpb ::= cluster_config
 *       | cluster_input
 *       | cluster_output
 *       | cpb
 *
 * cluster_proc ::= config_block
 *                | process_block
 *                | connect_block
 *                | cluster_proc
 */
sprokit::cluster_blocks
pipe_parser::
parse_cluster( std::istream& input )
{
  // There is only one cluster block per cluster.
  // cluster block "cluster"
  auto t = m_lexer.get_token();
  expect_token( TK_CLUSTER, t );

  cluster_pipe_block cpb;

  // Get cluster name as TK_IDENTIFIER
  if (t->token_type() != TK_IDENTIFIER )
  {
    PARSE_ERROR( t, "Expected cluster name but found " << t->text() );
  }

  cpb.type = t->text(); // save cluster name

  // get optional description
  cpb.description = collect_comments();

  while( true )
  {
    t = m_lexer.get_token();

    // cluster imap
    if ( t->token_type() == TK_IMAP )
    {
      m_lexer.unget_token( t );
      cluster_input_t imap;
      cluster_input( imap );
      cpb.subblocks.push_back( imap );
      continue;
    }

    // cluster omap
    if ( t->token_type() == TK_OMAP )
    {
      m_lexer.unget_token( t );
      cluster_output_t omap;
      cluster_output( omap );
      cpb.subblocks.push_back( omap );
      continue;
    }

    // cluster config entry
    // cluster_config_t contains only one config entry
    cluster_config_t cfg;
    if ( cluster_config( cfg ) )
    {
      cpb.subblocks.push_back( cfg );
      continue;
    }

    // unexpected token
    break;
  } // end while

  m_lexer.unget_token( t );

  // Add cluster header to the list.
  m_cluster_blocks.push_back( cpb );

  while (true )
  {
    t = m_lexer.get_token();

    // parse the process config and connect blocks
    if ( t->token_type() == TK_PROCESS )
    {
      m_lexer.unget_token( t );
      process_pipe_block ppb;
      process_definition( ppb );

      m_cluster_blocks.push_back( ppb );

      continue;
    }

    if ( t->token_type() == TK_CONFIG )
    {
      m_lexer.unget_token( t );
      config_pipe_block cpb;
      process_config_block( cpb );
      m_cluster_blocks.push_back( cpb );
      continue;
    }

    if ( t->token_type() == TK_CONNECT )
    {
      m_lexer.unget_token( t );
      connect_pipe_block cpb;
      process_connection( cpb );
      m_cluster_blocks.push_back( cpb );
      continue;
    }

    // unexpected token
    PARSE_ERROR( t, "Found unexpected token \"" << t->text() << "\"" );
  } // end while

  return m_cluster_blocks;
}


// ==================================================================
/**
 * @brief Handle process definition production
 *
 * "process" <proc_name> "::" <proc-type> <opt-config>
 */
void
pipe_parser::
process_definition(process_pipe_block& ppb)
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that his is a process token
  expect_token( TK_PROCESS, t );

  // Save location of the "process" keyword
  ppb.loc = t->get_location();

  t = m_lexer.get_token();      // get from process name
  if (t->token_type() != TK_IDENTIFIER)
  {
    PARSE_ERROR( t, "Expected process name but found " << t->text() );
  }

  ppb.name = t->token_value();

  t = m_lexer.get_token();      // get '::' separator
  if (t->token_type() != TK_DOUBLE_COLON)
  {
    PARSE_ERROR( t, "Expected process \"::\" but found " << t->text() );
  }

  t = m_lexer.get_token();      // get from process type
  if (t->token_type() != TK_IDENTIFIER)
  {
    PARSE_ERROR( t, "Expected process type but found " << t->text() );
  }

  ppb.type = t->token_value();

  // Handle the optional config lines
  parse_config( ppb.config_values );
}


// ------------------------------------------------------------------
/**
 * @brief Parse top level config block.
 *
 * "config" <config-key><EOL>
 * <config-entry-list>
 */
void
pipe_parser::
process_config_block( config_pipe_block& cpb )
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that this is a config token
  expect_token( TK_CONFIG, t );

  // Save location of the "config" keyword
  cpb.loc = t->get_location();

  // initiate reporting of EOL
  m_lexer.absorb_eol( false );

  // process the block name. Need to split up name into components
  while ( true )
  {
     t = m_lexer.get_token();

     if ( t->token_value() == TK_EOL)
     {
       break;
     }

    // expecting TK_IDENTIFIER ":" TK_IDENTIFIER... <eol>
     if ( t->token_type() == TK_IDENTIFIER )
     {
       cpb.key.push_back( t->text() );
     }

     t = m_lexer.get_token();
     if ( t->token_value() != ':')
     {
       PARSE_ERROR( t, "Expected ':' separating config key components, but \""
                  << t->text() << "\" found" );
     }
  } // end while

  m_lexer.absorb_eol( true );

  // process config lines in block
  parse_config( cpb.values );
}


// ------------------------------------------------------------------
/**
 * @brief Parse old style config entries.
 *
 * ":"<key><opt-attrs><whitespace><value>
 *
 * key ::= id : id : ...
 *
 * opt_attrs ::=
 *             | "[" <attr-list> "]"
 *
 * attr-list ::= attr
 *             | attr <attr_list>
 */
void
pipe_parser::
old_config( sprokit::config_value_t& val )
{
  // Note that the leading ':' has been absorbed

  // We need whitepsace tokens to parse the old format.
  m_lexer.absorb_whitespace( false );
  token_sptr t;

  // process the block name. Need to split up name into components
  while ( true )
  {
     t = m_lexer.get_token();

    // expecting TK_IDENTIFIER ":" ...
     if ( t->token_type() == TK_IDENTIFIER)
     {
       val.key.key_path.push_back( t->text() );
     }

     t = m_lexer.get_token();
     if ( t->token_value() != ':')
     {
       break;
     }
  } // end while

  // expecting '[' or something else (a space)
  if ( t->token_value() == '[' )
  {
    // process attrs
    parse_attrs( val );
  }

  t = m_lexer.get_token();

  // should be a space
  if ( t->token_type() != TK_WHITESPACE )
  {
    PARSE_ERROR( t, "Expected whitespace but found " << token::token_name( t->token_value() ) );
  }

  // save rest of line as the config value
  val.value = m_lexer.get_rest_of_line();

  m_lexer.absorb_whitespace( true );
}


// ------------------------------------------------------------------
/**
 * @brief Parse new style config entries.
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
pipe_parser::
new_config( sprokit::config_value_t& val )
{
  token_sptr t;

  // process the entry name. Need to split up name into components
  while ( true )
  {
     t = m_lexer.get_token();

     // expecting TK_IDENTIFIER ":" ...
     if ( t->token_type() == TK_IDENTIFIER)
     {
       val.key.key_path.push_back( t->text() );
     }
     else
     {
       PARSE_ERROR( t, "Expecting config key component but found \"" << t->text() << "\"" );
     }

     t = m_lexer.get_token();
     if ( t->token_value() != ':')
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
      val.key.options.flags->push_back( "local-assign" );
    }

    // save rest of line as the config value
    val.value = m_lexer.get_rest_of_line();
  }
  else
  {
    PARSE_ERROR( t, "Expecting assignment operator but found \"" << t->text() << "\"" );
  }
}


// ------------------------------------------------------------------
/**
 * @brief Parse attribute list.
 *
 * This method parses the attribute list. The leading '[' has already
 * been absorbed. All attributes up to the closing ']' are added to
 * the attribute list. The closing ']' is also absorbed.
 *
 * '[' <attr-list> ']'
 *
 * attr-list ::= attr
 *             | attr ',' attr_list
 *
 * @param[out] val Attributes are set in this parameter
 */
void
pipe_parser::
parse_attrs( sprokit::config_value_t& val )
{
  auto t = m_lexer.get_token();

  while( true )
  {
    if ( t->token_type() != TK_ATTRIBUTE )
    {
      PARSE_ERROR( t, "Expecting attribute name but found \"" << t->text() << "\"" );
    }

    val.key.options.flags->push_back( t->text() );

    t = m_lexer.get_token();

    if ( t->token_value() == ']' )
    {
      return;
    }

    if ( t->token_value() != ',' )
    {
      PARSE_ERROR( t, "Expecting ',' but found \"" << t->text() << "\"" );
    }
  } // end while
}


// ------------------------------------------------------------------
/**
 * Connection production
 *
 * "connect" "from" <proc>"."<port> "to" <proc>"."<port>
 */
void
pipe_parser::
process_connection( connect_pipe_block& cpb )
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
}


// ------------------------------------------------------------------
/**
 * @brief Parse cluster config entry
 *
 * This method processes a single cluster config entry.
 *
 * -- old style
 * :<key> <value>
 * :<key>[attr-list] <value>
 * "--" description.
 *
 * -- new style
 * <key> "=" <value>
 * <key>"["<attr-list"]" "=" <value>
 * "--" description.
 *
 * @return \b true if a config entry was parsed. \b false if not a
 * valid config entry. Note that a false return does not indicate an
 * error, just not a config entry.
 */
bool
pipe_parser::
cluster_config( cluster_config_t& cfg )
{

  if ( ! parse_config_line( cfg.config_value ) )
  {
    // error parsing
    return false;
  }

  // check for comments
  cfg.description = collect_comments();

  return true;
}


// ------------------------------------------------------------------
/**
 * @brief Parse cluster IMAP definition
 *
 * "imap" "from" <port> "to" <port_list> <EOL> <description>
 *
 * port_list ::= <port_spec>
 *             | <port_spec> , <port_list>
 *
 * port_spec ::= <process>"."<port>
 *
 * "--" description
 */
void
pipe_parser::
cluster_input( cluster_input_t& imap )
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that this is the correct token
  expect_token( TK_IMAP, t );

  // Validate noise word "from"
  t = m_lexer.get_token();
  expect_token( TK_FROM, t );

  // Get port name
  t = m_lexer.get_token();
  if (t->token_type() != TK_IDENTIFIER )
  {
    PARSE_ERROR( t, "Expected port name but found " << t->text() );
  }

  imap.from = t->text();

  // Validate noise word "to"
  t = m_lexer.get_token();
  expect_token( TK_TO, t );

  // initiate reporting of EOL
  m_lexer.absorb_eol( false );

  // PArse list of "to" port specs
  while ( true )
  {
    // parse port addr list
    process::port_addr_t port_addr;
    parse_port_addr( port_addr );

    // Add to list of ports
    imap.targets.push_back( port_addr );

    t = m_lexer.get_token();

    // end of line is end of statement
    if ( t->token_type() == TK_EOL )
    {
      break;
    }

    // If this is a comma, then
    if ( t->token_value() == ',' )
    {
      t = m_lexer.get_token();
    }
    else
    {
      PARSE_ERROR( t, "Expecting comma or EOL but found \"" << t->text() << "\"" );
    }
  } // end while

  m_lexer.absorb_eol( true );

  imap.description = collect_comments();
}


// ------------------------------------------------------------------
/**
 * @brief Parse cluster OMAP definition
 *
 * "omap" "from" <process>"."<port> "to" <port>
 * "--" description
 */
void
pipe_parser::
cluster_output( cluster_output_t& omap )
{
  auto t = m_lexer.get_token();

  // Should be guaranteed that this is the correct token
  expect_token( TK_IMAP, t );

  // Validate noise word "from"
  t = m_lexer.get_token();
  expect_token( TK_FROM, t );

  parse_port_addr( omap.from );

  // Validate noise word "to"
  t = m_lexer.get_token();
  expect_token( TK_TO, t );

  // Get port name
  t = m_lexer.get_token();
  if (t->token_type() != TK_IDENTIFIER )
  {
    PARSE_ERROR( t, "Expected port name but found " << t->text() );
  }

  omap.to = t->text();

  omap.description = collect_comments();
}


// ------------------------------------------------------------------
/**
 * @brief Collect sequential cluster comment lines.
 *
 * Cluster comments start with "--" string, sequential lines that are
 * comments will be collected. Each line is ended with a new-line character.
 *
 * @return Collected comment or empty string if no comments found.
 */
std::string
pipe_parser::
collect_comments()
{
  token_sptr t;
  std::string comments;

  while (true )
  {
    t = m_lexer.get_token();
    if ( t->token_type() != TK_CLUSTER_DESC )
    {
      break;
    }

    comments += m_lexer.get_rest_of_line() + "\n";
  } // end while

  m_lexer.unget_token( t );

  return comments;
}


// ------------------------------------------------------------------
/**
 * @brief Parse port addr specification.
 *
 * port_addr::= proc_name '.' port_name
 *
 * @param[out] out_pa The port address parts are returned hjere.
 */
void
pipe_parser::
parse_port_addr( process::port_addr_t& out_pa)
{
  auto t = m_lexer.get_token();      // get from process name
  if (t->token_type() != TK_IDENTIFIER )
  {
    PARSE_ERROR( t, "Expected process name but found " << t->text() );
  }

  const std::string proc_name( t->text() );

  t = m_lexer.get_token();      // get separator
  expect_token( '.', t );

  t = m_lexer.get_token();      // get from port name
  if (t->token_type() != TK_IDENTIFIER )
  {
    PARSE_ERROR( t, "Expected port name but found " << t->text() );
  }

  const std::string port_name( t->text() );

  // copy output to parameter
  out_pa = process::port_addr_t( proc_name, port_name );
}


// ------------------------------------------------------------------
/**
 * @brief Parse series of config lines.
 *
 * This method parses zero or more sequential config
 * specifications. Both new and old style config entries are handled.
 *
 * @param[out] out_config Vector updated with config entries parsed.
 */
void
pipe_parser::
parse_config( config_values_t& out_config )
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

    if ( t->token_type() == TK_BLOCK )
    {
      // handle block name

      const std::string block_name = m_lexer.get_rest_of_line();

      // Save current block context and start another
      block_context_t block_ctxt;
      block_ctxt.m_block_name = block_name; // block name
      block_ctxt.m_location = m_lexer.current_location();
      block_ctxt.m_previous_context = current_context;

      kwiver::vital::tokenize( block_name, current_context, ":", true );

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
    config_val.key.key_path = current_context;

    if ( ! parse_config_line( config_val ) )
    {
      // not a valid config line. Could be something else valid though.
      break;
    }

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

    throw parsing_exception( msg.str() );
  }
}


// ------------------------------------------------------------------
/**
 * @brief Process a single config line.
 *
 * This method processes a single config line of either the old or new
 * style. If the config line is valid, the output parameter is updated.
 *
 * @param[out] config_val Updated with config information.
 *
 * @return \b true if valid config line processed. This means that the
 * return parameter is valid. \b false indicates that the line was not
 * a valid config line and the output parameter is not valid.
 */
bool
pipe_parser::
parse_config_line( config_value_t& config_val )
{
  bool ret_status(true);        // assume o.k.
  auto t = m_lexer.get_token();

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
    config_val.key.options.flags->push_back( "relativepath" );

    // get next token
    t = m_lexer.get_token();
  }

  // possibly a new style config <key> "=" <value>
  // possibly a new style config <key>"["<attr-list"]" "=" <value>
  auto lat = m_lexer.get_token();
  if ( ( lat->token_type() == TK_ASSIGN )
       || ( lat->token_type() == TK_LOCAL_ASSIGN )
       || ( lat->token_value() == '[' ) ) // possible attributes
  {
    // push last two tokens back to lexer
    m_lexer.unget_token( lat );
    m_lexer.unget_token( t );

    // process new style config
    new_config( config_val );

    return ret_status;
  }
  else
  {
    // probably not an config statement
    // push last two tokens back to lexer
    m_lexer.unget_token( lat );
    m_lexer.unget_token( t );
  }

  // indicate some other type of token
  return false;
}




// ------------------------------------------------------------------
bool
pipe_parser::
expect_token( int expected_tk, token_sptr t )
{
  if ( t->token_type() != expected_tk )
  {
    PARSE_ERROR( t, "Expected \"" << token::token_name( expected_tk ) << "\" keyword, but found "
               << t->text() );
  }

  return true;
}

} // end namespace
