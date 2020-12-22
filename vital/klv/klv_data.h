// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface for the klv data class.
 */

#ifndef KWIVER_VITAL_KLV_DATA_H_
#define KWIVER_VITAL_KLV_DATA_H_

#include <vital/klv/vital_klv_export.h>

#include <cstddef>
#include <iostream>
#include <vector>
#include <cstdint>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** A container for a raw KLV packet.
 *
 * This class represents a single KLV packet, that includes Key,
 * Length, and Value components.
 *
 * An object of this class is immutable. Once it is created, it can
 * not be changed, only querried.
 */
class VITAL_KLV_EXPORT klv_data
{
public:
  typedef std::vector< uint8_t > container_t;
  typedef container_t::const_iterator const_iterator_t;

  klv_data();

  /** Build a new object from raw packet. A raw packet is supplied
   * with the offsets to the intresting parts. The raw data is copied
   * into this object so the caller can dispose of its copy as
   * desired.
   *
   * What's not captured is the length of the length, but that can be
   * determined if needed.
   */
  klv_data(container_t const& raw_packet,
           size_t key_offset, size_t key_len,
           size_t m_value_offset, size_t value_len);

  ~klv_data();

  /// The number of bytes in the key
  std::size_t key_size() const;

  /// Number of bytes in the value portion
  std::size_t value_size() const;

  /// Number of bytes in whole packet
  std::size_t klv_size() const;

  /// Iterators for raw packet
  const_iterator_t klv_begin() const;
  const_iterator_t klv_end() const;

  /// Iterators for key
  const_iterator_t key_begin() const;
  const_iterator_t key_end() const;

  /// Iterators for value
  const_iterator_t value_begin() const;
  const_iterator_t value_end() const;

private:
  std::vector< uint8_t > m_raw_data;
  std::size_t m_key_offset;
  std::size_t key_len_;
  std::size_t m_value_offset;
  std::size_t m_value_len;
};

/// Output operator
std::ostream& operator<< (std::ostream& str, klv_data const& obj);

} } // end namespace

#endif
