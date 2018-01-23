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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * @file
 * @brief Various utility functions for parsing KPF.
 *
 */

#ifndef KWIVER_VITAL_KPR_PARSE_UTILS_H_
#define KWIVER_VITAL_KPR_PARSE_UTILS_H_



#include <arrows/kpf/yaml/kpf_packet.h>

#include <string>
#include <vector>
#include <tuple>
#include <map>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * @brief This maps KPF packet headers to their full packets.
 *
 * The packet buffer holds the parsed KPF packets for the current line (aka record.)
 * Packets are transferred out of the buffer to the client via the kfp_reader.
 *
 * The packet buffer is a multimap because some packets may appear multiple
 * times (i.e. key-value packets.)
 */

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp > packet_buffer_t;

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp >::const_iterator packet_buffer_cit;


typedef std::tuple< bool, std::string, int > header_parse_t;

/**
 * @brief Convert a string (e.g. 'id2') into a KPF packet header.
 *
 * @return true if the conversion is successful.
 */

packet_header_t KPF_YAML_EXPORT
packet_header_parser( const std::string& s );

header_parse_t KPF_YAML_EXPORT parse_header( const std::string& s, bool expect_colon );


} // ...kpf
} // ...vital
} // ...kwiver

#endif
