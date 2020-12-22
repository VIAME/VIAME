// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
