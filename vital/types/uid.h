/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
