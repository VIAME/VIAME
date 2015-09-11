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
 * This class converts config file contents into a config block.  The
 * intent is that this parser converts one input file into one config
 * block.  Delete the object when done. Allocate another if another
 * file needs to be processed.
 *
 */
class config_parser
  : private kwiver::vital::noncopyable
{
public:

  /**
   * \brief Create object
   *
   * \param file_path Name of file to parse and convert to config block.
   *
   */
  config_parser( config_path_t const& file_path );
  ~config_parser();


  /**
   * \brief Parse file into a config block
   *
   * The file specified by the CTOR is read and parsed
   *
   * \throws config_file_not_parsed_exception
   *
   * \throws config_file_not_found_exception
   */
  void parse_config();

  /**
   * \brief Get processed config block.
   *
   * This method returns a sptr to the processed config block.
   *
   * \return Pointer to config block
   */
  kwiver::vital::config_block_sptr get_config() const;

  // method to add token classes

private:

  class priv;

  config_path_t m_config_file;
  std::auto_ptr< priv > m_priv;
};

} } // end namespace

#endif /* KWIVER_VITAL_CONFIG_PARSER_H */
