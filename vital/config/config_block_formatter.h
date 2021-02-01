/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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


#ifndef KWIVER_CONFIG_FORMATTER_H
#define KWIVER_CONFIG_FORMATTER_H

#include <vital/config/vital_config_export.h>
#include <vital/config/config_block.h>

#include <string>
#include <ostream>

namespace kwiver {
namespace vital {

/**
 * @brief Generates formatted versions of a config block.
 *
 * This class encapsulates several different formatting options for
 * a config block.
 */
class VITAL_CONFIG_EXPORT config_block_formatter
{
public:
  config_block_formatter( const config_block_sptr config );
  ~config_block_formatter() = default;

  /**
   * @brief Format config block in simple text format.
   *
   * @param str Stream to format on.
   */
  void print( std::ostream& str );

  /**
   * @brief Set line prefix for printing.
   *
   * @param pfx The prefix string.
   */
  void set_prefix( const std::string& pfx );

  /**
   * @brief Set option to generate source location.
   *
   * @param opt TRUE will generate the source location, FALSE will not.
   */
  void generate_source_loc( bool opt );

private:
  void format_block( std::ostream& str,
                     const config_block_sptr config,
                     const std::string& prefix );

  config_block_sptr m_config;
  std::string m_prefix;
  bool m_gen_source_loc;
};

} } // end namespace

#endif /* KWIVER_CONFIG_FORMATTER_H */
