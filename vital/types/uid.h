// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to vital global uid
 */

#ifndef KWIVER_VITAL_TYPES_UID_H
#define KWIVER_VITAL_TYPES_UID_H

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <string>
#include <cstdint>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Container for global uid value
 *
 * This class represents a global UID. The content and other
 * attributes are dependent on the method used to create the ID.
 */
class VITAL_EXPORT uid
{
public:
  //@{
  /**
   * @brief Create uid
   *
   * This constructor creates an object with a specific ID that may
   * not be globally unique. That is up to the caller. If a really
   * globally unique id, of standard form, is needed, use the factory
   * function.
   *
   * This method allows the caller to create an ID for specific uses
   * by customizing the content.
   *
   * @param data Byte array to be held as uid
   */
  uid( const std::string& data );
  uid( const char* data, size_t byte_count );
  //@}

  uid();

  ~uid() = default;

  /**
   * @brief Report if this uuid is valid.
   *
   * @return \b true if uid is valid; false otherwise.
   */
  bool is_valid() const;

  /**
   * @brief Return uid value
   *
   * This method returns a pointer to the actual list of bytes that
   * make up the ID.
   *
   * @return pointer to the data bytes.
   */
  std::string const& value() const;

  /**
   * @brief Get number of bytes in id
   *
   *
   * @return Number of bytes.
   */
  size_t size() const;

  /// equality operator
  bool operator==( const uid& other ) const;
  bool operator!=( const uid& other ) const;
  bool operator< ( const uid& other ) const;

private:
  std::string  m_uid;

}; // end class uid

} } // end namespace

#endif // KWIVER_VITAL_TYPES_UID_H
