// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_KLV_MISP_TIME_H
#define VITAL_KLV_MISP_TIME_H

#include <vital/klv/vital_klv_export.h>

#include <vector>
#include <cstdint>

namespace kwiver {
namespace vital {

/**
 * @brief Find MISP time packet in raw buffer and convert.
 *
 * This function scans the supplied raw video metadata buffer to see
 * if it contains a MISP time packet. If it does, the packet is
 * converted to a time value and returned.
 *
 * @param[in] raw_data Raw metadatadata buffer
 * @param[out] ts Time from MISP packet.
 *
 * @return \b true if MISP time packet found in buffer.
 */
VITAL_KLV_EXPORT
bool find_MISP_microsec_time(  std::vector< unsigned char > const& raw_data, std::int64_t& ts );

/**
 * @brief Convert MISP time packet to uSec
 *
 * This function converts the supplied MIST time packet to
 * microseconds.
 *
 * @param[in] buf Raw packet to convert
 * @param[out] ts Time from packet in microseconds
 *
 * @return \b true if the buffer passes validity checks.
 */
VITAL_KLV_EXPORT
bool convert_MISP_microsec_time( std::vector< unsigned char > const& buf, std::int64_t& ts );

} } // end namespace

#endif /* VITAL_KLV_MISP_TIME_H */
