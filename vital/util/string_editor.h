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

#ifndef VITAL_STRING_EDITOR_H
#define VITAL_STRING_EDITOR_H

#include <vital/util/vital_util_export.h>
#include <vital/util/string.h>

#include <string>
#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Editing operations on string.
 *
 * This class is the abstract interface for a string edit operation.
 * A string is presented to the process() method which can return a
 * modified string or absorb the string.
 */
class VITAL_UTIL_EXPORT string_edit_operation
{
public:
  string_edit_operation() { }
  virtual ~string_edit_operation() { }

  /**
   * @brief Edit the string.
   *
   * This method is called with a string to be processed. The string can
   * be returned unmodified, modified, or not returned at all.
   *
   * @param[in,out] line The line to be processed without trailing new line
   *
   * @return \b true if the edited string is returned through the
   * parameter. \b false if the string has been absorbed and there is
   * no data being returned.
   */
  virtual bool process( std::string& line ) = 0;
};


// ----------------------------------------------------------------
/**
 * @brief Apply editing operations to a string.
 *
 * This class represents a generic set of string editing operations
 * applied in a fixed order. An instance of this class is configured
 * by adding one or more string_edit_operation objects.
 */
class VITAL_UTIL_EXPORT string_editor
{
public:
  // -- CONSTRUCTORS --
  string_editor();
  virtual ~string_editor();

  /**
   * @brief Append an edit operation to the end of the list.
   *
   * This method appends the specified edit operation to the end of
   * the end of the current list of edit operations.
   *
   * This object takes ownership of the parameter object.
   *
   * @param op New exit operation to add to list.
   */
  void add( string_edit_operation* op );

  /**
   * @brief Apply editors to the string.
   *
   * This method applies all registered editors to the string, passing
   * the output of one editor to the next.
   *
   * @param str The string to be edited.
   *
   * @return \b true if the line has been edited. \b false if the line has been absorbed.
   */
  bool edit( std::string& str );

private:
  std::vector< std::shared_ptr< string_edit_operation > > m_editor_list;
};// end class string_editor


namespace edit_operation {

// ==================================================================
// Some commonly used editing operations.
//
/**
 * @brief Remove shell comments
 *
 * This class removes standard shell comments from the supplied
 * string. A shell comment is defined as the substring that starts
 * with the character '#' and continues to the end of the string. Note
 * that that trailing white space is not removed.
 */
class shell_comment : public string_edit_operation
{
public:
  virtual bool process( std::string& line )
  {
    auto pos = line.find_first_of( "#" );

    if ( pos != std::string::npos )
    {
      line.erase( pos );
    }

    return true;
  }
};


// ------------------------------------------------------------------
/**
 * @brief Absorb blank lines.
 *
 * This class removes strings that are all blank. If the string is all
 * blank, then it is absorbed.
 *
 */
class remove_blank_string : public string_edit_operation
{
public:
  virtual bool process( std::string& str )
  {
    if ( str.find_first_not_of( " \t\n\r\f\v" ) != std::string::npos )
    {
      return true;
    }

    return false;
  }
};


// ----------------------------------------------------------------
/**
 * @brief Remove leading whitespace.
 *
 * This class removes whitespace from the left or leading end of the
 * string.
 */
class left_trim : public string_edit_operation
{
public:
  virtual bool process( std::string& s )
  {
    kwiver::vital::left_trim(s);
    return true;
  }
};   // end class left_trim


// ----------------------------------------------------------------
/**
 * @brief Remove trailing whitespace.
 *
 * This class removes trailing whitespace from the right side of the
 * string.
 */
class right_trim : public string_edit_operation
{
public:
  virtual bool process( std::string& s )
  {
    kwiver::vital::right_trim(s);
    return true;
  }
}; // end class right_trim

} // end namespace edit_operation

} } // end namespace

#endif /* VITAL_STRING_EDITOR_H */
