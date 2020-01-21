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
 * @brief Support for predefined KPF yaml schemas.
 *
 * The KPF schema is mostly a validation aid, associating a tag ("geom",
 * "act") with a set of required packet types.
 *
 */

#ifndef KWIVER_VITAL_KPF_YAML_SCHEMAS_H_
#define KWIVER_VITAL_KPF_YAML_SCHEMAS_H_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace kwiver {
namespace vital {
namespace kpf {

enum class schema_style {
  INVALID,     // invalid
  META,        // metadata
  GEOM,        // geometry
  ACT,         // activity
  TYPES,       // object types
  REGIONS,     // regions file
  UNSPECIFIED  // no validation provided
};

struct KPF_YAML_EXPORT validation_data
{
  packet_style style;
  std::string key;
  validation_data(): style( packet_style::INVALID ), key("") {}
  explicit validation_data( YAML::const_iterator it );
  explicit validation_data( packet_style s ): style(s), key("") {}
  validation_data( packet_style s, const std::string& k ): style(s), key(k) {}

  static std::string schema_style_to_str( schema_style s );
  static schema_style str_to_schema_style( const std::string& s );
};

std::vector< validation_data > KPF_YAML_EXPORT
validate_schema( schema_style schema, const packet_buffer_t& packets );

std::vector< validation_data > KPF_YAML_EXPORT
validate_schema( schema_style schema, const std::vector< validation_data>& vpackets );


} // ...kpf
} // ...vital
} // ...kwiver

#endif
