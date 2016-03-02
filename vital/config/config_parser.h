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

#ifndef KWIVER_VITAL_CONFIG_PARSER_H
#define KWIVER_VITAL_CONFIG_PARSER_H

#include <vital/config/config_block.h>

#include <vital/noncopyable.h>
#include <memory>


namespace kwiver {
namespace vital {


// ----------------------------------------------------------------
/**
 * \brief Config file parser.
 *
 * This class converts config file contents into a single config
 * block.  The intent is that this parser creates one config block
 * from desired inputs.  Delete the object when done. Allocate another
 * if another config block is needed.
 *
 * Files specified on the include directive are found by scanning the
 * search path. If they are not found on that path then the directory
 * of the config file that contains the include directive is checked.
 *
 * The file passed to the parse_config() directory are taken as is and
 * not searched for in the path. This file should have already been
 * located by the calling logic.
 *
 * Example code:
\code
std::auto< kwiver::vital::config_parser> input_config( new kwiver::vital::config_parser() );
input_config->add_search_path( "../config" );
input_config->parse_config( filename );
// Optionally, additional files can be parsed into the same config block.
kwiver::vital::config_block_sptr blk = input_config->get_config();
\endcode
 *
 */
class config_parser
  : private kwiver::vital::noncopyable
{
public:

  /**
   * \brief Create object
   *
   */
  config_parser();
  ~config_parser();

//@{
  /**
   * \brief Add directory to search path.
   *
   * This method adds a directory to the end of the config file search
   * path. This search path is used to locate all referenced included
   * files.
   *
   * @param file_path Directory or list to add to end of search path.
   */
  void add_search_path( config_path_t const& file_path );
  void add_search_path( config_path_list_t const& file_path );
//@}


  /**
   * \brief Get config path list.
   *
   * This method returns the list of directories that are searched
   * when looking for config files.
   *
   * \return List of directories that make up the search path.
   */
  config_path_list_t const& get_search_path() const;


  /**
   * \brief Parse file into a config block
   *
   * The file specified is read and parsed. The resulting config block
   * is available via the get_config() method. Additional files can be
   * parsed by this method and have their contents aggregated in the
   * same config block.
   *
   * The file specified is taken as supplied and an attempt is made to
   * read it. Files on include directives are resolved against the
   * search path if they are a relative file path.  If a file can not
   * be found in the current search path, then the directory of the
   * config file containing the include directive is checked.
   *
   * \param file_path Name of file to parse and convert to config block.
   *
   * \throws config_file_not_parsed_exception
   *
   * \throws config_file_not_found_exception
   */
  void parse_config( config_path_t const& file_path );

  /**
   * \brief Get processed config block.
   *
   * This method returns a sptr to the processed config block. This
   * block can be the combined contents of several files.
   *
   * \return Pointer to config block
   */
  kwiver::vital::config_block_sptr get_config() const;

protected:
  // method to add token classes

private:

  class priv;

  config_path_t m_config_file;
  const std::unique_ptr< priv > m_priv;
};

} } // end namespace

#endif /* KWIVER_VITAL_CONFIG_PARSER_H */
