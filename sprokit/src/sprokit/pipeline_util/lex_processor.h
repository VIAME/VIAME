/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_LEX_PROCESS_H
#define SPROKIT_PIPELINE_LEX_PROCESS_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "token.h"

#include <vital/vital_config.h>
#include <vital/logger/logger.h>
#include <vital/util/source_location.h>
#include <vital/config/config_block_types.h>

#include <string>
#include <memory>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief lexical processor for pipeline files.
 *
 * This class reads an input file and returns a set of tokens, one
 * token at a time.
 *
 * USAGE:
 * 1) create lex_processor;
 * 2) configure search paths.
 * 3) call open_file(filename)
 * 4) call get_token() to get next token
 * 5) Quit when EOF token is returned.
 *
 */
class SPROKIT_PIPELINE_UTIL_EXPORT lex_processor final
{
public:
  /**
   * @brief Constructor.
   *
   * Include file search paths can be added after the CTOR call.
   *
   */
  lex_processor();
  ~lex_processor();

  /**
   * @brief Open specified file for tokenizing
   *
   * This method open the specified file for tokenizing. The name of
   * the file must be such that it can be directly opened since the
   * search paths are not used for the initial file.
   *
   * No tokens are processed by this method. Call get_token() for
   * the next token.
   *
   * @param file_name Name of file to open.
   *
   * @throws kwiver::vital::file_not_found_exception
   */
  void open_file( const std::string& file_name );

  /**
   * @brief Read tokens from input stream.
   *
   * @param input Stream to read text from.
   * @param name Name of stream. This could be the name of the file or a
   * marker string for a stream with another source.
   */
  void open_stream( std::istream& input, const std::string& name = "" );

  /**
   * @brief Get source location for current position in input.
   *
   * This method returns the file name and line number where the lex
   * process is currently working. Calling get_next_token() will
   * advance this location past the token returned. If a token is
   * pushed back, this source location is not updated.
   *
   * @return Source location object indicating current file and line
   * number being processed.
   */
  kwiver::vital::source_location current_location() const;

  /**
   * @brief Return next token from input.
   *
   * This method returns the next token from the input stream or the
   * token from the top of the token push-back stack if the stack is
   * not empty.
   *
   * @return Object representing the next token taken taken from the
   * input stream.
   */
  token_sptr get_token();

  /**
   * @brief Push token back to lex processor.
   *
   * This method pushed the specified token onto an internal
   * stack. Tokens pushed to this stack will be returned by
   * get_next_token() in preference to new tokens taken from the input
   * stream.
   *
   * @param token Token to push back.
   */
  void unget_token( token_sptr token );

  /**
   * @brief Return remainder of input line.
   *
   * This method returns the remaining characters in the current
   * line. The terminating EOL is consumed, but not returned.
   *
   * This method is designed to assist in processing configuration
   * specifications where the remainder of the line is to be used as
   * the value for a config entry.
   *
   * @return Remainder of input line.
   */
  std::string get_rest_of_line();

  /**
   * @brief Flush remainder of line in parser.
   *
   * This method skips to the next character after the next new-line.
   */
  void flush_line();

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
   * @brief Set mode to absorb EOL or not.
   *
   * This option is set based on the parser state to enable or disable
   * reporting of EOL tokens. Generally they are not reported but, due
   * to quirks in the original syntax, this is helpful.
   *
   * @param opt \b true indicates that EOL is not to be reported.
   */
  void absorb_eol( bool opt);

  /**
   * @brief Set mode to absorb whitespace.
   *
   * This option is set based on the parser state to enable or disable
   * reporting of whitespace tokens. Generally they are not reported
   * but, due to quirks in the original syntax, this is helpful.
   *
   * @param opt \b true indicates that whitespace is not to be reported.
   */
  void absorb_whitespace( bool opt);

private:
  token_sptr get_next_token();
  bool get_next_line();

  kwiver::vital::logger_handle_t m_logger;
  kwiver::vital::config_path_t m_config_file;

  class priv;
  std::unique_ptr< priv > m_priv;
};

} // end namespace

#endif /* SPROKIT_PIPELINE_LEX_PROCESS_H */
