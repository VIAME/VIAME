/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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

/** @file
 * Interface to the KLV parsing functions.
 */

#ifndef KWIVER_VITAL_KLV_PARSE_H_
#define KWIVER_VITAL_KLV_PARSE_H_

#include <vital/klv/vital_klv_export.h>
#include <vital/klv/klv_key.h>

#include <vector>
#include <deque>
#include <ostream>
#include <cstdint>

namespace kwiver {
namespace vital {

class klv_data;

/// Define a type for KLV LDS key-value pairs
typedef std::pair<klv_lds_key, std::vector<uint8_t> > klv_lds_pair;
typedef std::vector< klv_lds_pair > klv_lds_vector_t;

/// Define a type for KLV UDS key-value pairs
typedef std::pair<klv_uds_key, std::vector<uint8_t> > klv_uds_pair;
typedef std::vector< klv_uds_pair > klv_uds_vector_t;


/**
 * @brief Pop the first KLV UDS key-value pair found in the data buffer.
 *
 * The first valid KLV packet found in the data stream is returned.
 * Leading bytes that do not belong to a KLV pair are dropped. If
 * there is a partial packet in the input data stream, it is left
 * there and no packet is returned.
 *
 * @param[in,out] data Byte stream to be parsed.
 * @param[out] klv_packet Full klv packet with key and data fields
 * specified.
 *
 * @return \c true if packet returned; \c false if no packet returned.
 */
VITAL_KLV_EXPORT bool
klv_pop_next_packet( std::deque< uint8_t >& data, klv_data& klv_packet);


/**
 * @brief Parse KLV LDS (Local Data Set) from an array of bytes.
 *
 * The input array is the raw KLV packet. The output is a vector of
 * LDS packets. The raw packet is usually taken from the
 * klv_pop_next_packet() function.
 *
 * @param data KLV raw packet
 *
 * @return A vector of klv LDS packets.
 */
VITAL_KLV_EXPORT klv_lds_vector_t
parse_klv_lds(klv_data const& data);


/**
 * @brief Parse KLV UDS (Universal Data Set) from an array of bytes.
 *
 * The input array is the raw KLV packet. The output is a vector of
 * UDS packets. The raw packet is usually taken from the
 * klv_pop_next_packet() function.
 *
 * The UDS keys can be decoded using the klv_0104 class.
 *
 * @param[in] data KLV raw packet
 *
 * @return A vector of klv UDS packets.
 */
VITAL_KLV_EXPORT klv_uds_vector_t
parse_klv_uds( klv_data const& data );


/**
 * @brief Print KLV packet.
 *
 * The supplied KLV packet is decoded and printed. The raw packet is
 * usually taken from the klv_pop_next_packet() function.
 *
 * @param str stream to format on
 * @param klv packet to decode
 */
VITAL_KLV_EXPORT std::ostream&
print_klv( std::ostream& str, klv_data const& klv );

} } // end namespace

#endif
