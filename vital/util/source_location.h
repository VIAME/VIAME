/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief Interface of source_location class.
 */

#ifndef KWIVER_VITAL_UTIL_SOURCE_LOCATION_H
#define KWIVER_VITAL_UTIL_SOURCE_LOCATION_H

#include <vital/util/vital_util_export.h>

#include <ostream>
#include <string>
#include <memory>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Location in a source file.
 *
 * This class represents a location in a source file.
 *
 * File names are expected as shared pointers in an effort to save
 * space. The expectation is that the file name will be used multiple
 * times.
 *
 * There are some cases where this class is used but not set. Use the
 * valid() method to determine it there is a real location specified.
 */
class VITAL_UTIL_EXPORT source_location
{
public:
  source_location();

  /**
   * @brief Create new source location from file and line number.
   *
   * @param f Shared pointer to file name string.
   * @param l Line number of definition.
   */
  explicit source_location( std::shared_ptr< std::string > f, int l );
  source_location( const source_location& other );
  virtual ~source_location();

  /**
   * @brief Generate formatted string for source location.
   *
   * @param str Stream to format on.
   *
   * @return Stream
   */
  virtual std::ostream& format( std::ostream& str ) const;

  /**
   * @brief Get file name
   *
   * This method returns the file name portion of the source location.
   *
   * @return File name string.
   */
  std::string const& file() const { return *m_file_name; }

  /**
   * @brief Get line number.
   *
   * This method returns the line number from the object.
   *
   * @return Line number of definition.
   */
  int line() const { return m_line_num; }

  /**
   * @brief Determine of object has valid data.
   *
   * Sometimes it is necessary to create an object of this type that
   * is not yet initialized or in cases where the location can not be
   * determined. This method determines if the object contains a valid
   * location.
   *
   * @return \b true if location is valid.
   */
  bool valid() const;

private:
  std::shared_ptr< std::string > m_file_name;
  int m_line_num;
};

inline std::ostream&
operator<<( std::ostream& str, source_location const& obj )
{ return obj.format( str ); }

} } // end namespace

#endif // KWIVER_VITAL_UTIL_SOURCE_LOCATION_H
