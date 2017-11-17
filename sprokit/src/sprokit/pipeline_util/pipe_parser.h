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
 * \brief Interface to pipeline parser.
 */

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_PARSER_H
#define SPROKIT_PIPELINE_UTIL_PIPE_PARSER_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "pipe_declaration_types.h"
#include "token.h"
#include "lex_processor.h"

#include <sprokit/pipeline/types.h>

#include <vital/vital_config.h>
#include <vital/vital_types.h>
#include <vital/logger/logger.h>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Pipe and cluster parser.
 *
 */
class SPROKIT_PIPELINE_UTIL_EXPORT pipe_parser final
{
public:
  // -- CONSTRUCTORS --
  pipe_parser();
  ~pipe_parser() = default;

  //@{
  /**
   * \brief Add directory to search path.
   *
   * This method adds a directory to the end of the config file search
   * path. This search path is used to locate all referenced included
   * files only.
   *
   * @param file_path Directory or list to add to end of search path.
   */
  void add_search_path( kwiver::vital::config_path_t const& file_path );
  void add_search_path( kwiver::vital::config_path_list_t const& file_path );
  //@}


  /**
   * @brief Parse a pipeline definition.
   *
   * Parse a pipeline definition file into pipe_blocks.
   *
   * @param input Stream to read pipeline definition from.
   * @param name Input file name
   *
   * @return A vector of pipe blocks representing the pipeline.
   */
  sprokit::pipe_blocks parse_pipeline( std::istream& input, const std::string& name = "" );

  /**
   * @brief Parse cluster definitions.
   *
   * Parse a cluster definition into the internal representation.
   *
   * @param input Stream to read cluster definitions.
   * @param name Input file name
   *
   * @return A vector of cluster blocks representing the cluster definition.
   */
  sprokit::cluster_blocks parse_cluster( std::istream& input, const std::string& name = "" );

  /** Compatibility mode.
   *
   * Indicates the current approach to handling old style pipe file
   * constructs.
   *
   * ALLOW - allows the old style without comment.
   *
   * WARN - issues a warning when old style constructs are
   *        encountered. This is useful when trying to update pipe
   *        files.
   *
   * ERROR - causes an error when old style constructs are
   *         encountered. This is useful when validating there are no
   *         old style constructs in the input.
   */
  enum compatibility_mode_t
  { COMPATIBILITY_ALLOW,
    COMPATIBILITY_WARN,
    COMPATIBILITY_ERROR };

  /**
   * @brief Set compatibility mode.
   *
   * This method sets the compatibility mode to use when encountering
   * old style pipeline constructs.
   *
   * @param mode Compatibility mode to use.
   */
  void set_compatibility_mode( compatibility_mode_t mode );


private:

  // production methods
  void process_definition( process_pipe_block& ppb );
  void process_config_block( config_pipe_block& pcb );
  void process_connection( connect_pipe_block& cpb );

  // Cluster productions
  void parse_one_cluster();
  bool cluster_config( cluster_config_t& cfg );
  void cluster_input( cluster_input_t& imap );
  void cluster_output( cluster_output_t& omap );

  // Support methods
  void parse_port_addr( process::port_addr_t& out_pa );
  void parse_config( config_values_t& out_config );
  bool parse_config_line( config_value_t& config_val );
  void old_config( sprokit::config_value_t& val );
  void new_config( sprokit::config_value_t& val );
  std::string collect_comments();
  void parse_attrs( sprokit::config_value_t& val );

  std::string parse_config_key();
  std::string parse_process_name();
  std::string parse_port_name();
  std::string parse_extended_id( const std::string& extra_char,  const std::string& expecting);

  bool expect_token( int expected_tk, token_sptr t );

  /// Current compatibility mode.
  compatibility_mode_t m_compatibility_mode;

  // root of the pipeline AST
  sprokit::pipe_blocks m_pipe_blocks;

  // root of the cluster AST
  sprokit::cluster_blocks m_cluster_blocks;

  kwiver::vital::logger_handle_t m_logger;

  sprokit::lex_processor m_lexer;

}; // end class pipe_parser

} // end namespace

#endif /* SPROKIT_PIPELINE_UTIL_PIPE_PARSER_H */
