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
