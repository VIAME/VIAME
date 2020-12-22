// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
