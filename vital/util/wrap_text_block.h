/*ckwg +29

 * Copyright 2016 by Kitware, Inc.
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

#ifndef VITAL_UTIL_WRAP_TEXT_BLOCK_H
#define VITAL_UTIL_WRAP_TEXT_BLOCK_H

#include <vital/util/vital_util_export.h>

#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Format long text into wrapped text block.
 *
 * This class formats a long text string into a more compact text block.
 */
class VITAL_UTIL_EXPORT wrap_text_block
{
public:
  wrap_text_block();
  virtual ~wrap_text_block();

  /**
   * @brief Set leading indent string.
   *
   * This method sets the indent string that is prepended to each
   * output line. This string can be a comment marker or a set of
   * blanks to indent the output.
   *
   * @param indent String to prepend to each output line.
   */
  void set_indent_string( const std::string& indent );

  /**
   * @brief Set output line length.
   *
   * This method sets the length on the output line.
   *
   * @param len Length of the output line.
   */
  void set_line_length( size_t len );

  /**
   * @brief Wrap text string.
   *
   * This method wraps the input string to the specified line
   * length. Existing newline characters are retained so user
   * specified line breaks are preserved. Multiple spaces are retained
   * to preserve user formatting. The returned string always ends with
   * a newline.
   *
   * @param text The text to be wrapped
   *
   * @return Wrapped string.
   */
  virtual std::string wrap_text( const std::string& text );

private:
  std::string m_indent;
  size_t m_line_length;

}; // end class wrap_text_block

} } // end namespace

#endif // VITAL_UTIL_WRAP_TEXT_BLOCK_H
