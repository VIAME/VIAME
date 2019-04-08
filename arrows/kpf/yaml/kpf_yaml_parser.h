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
 * @brief KPF YAML parser class.
 *
 * Header for the KPF YAML parser; holds the YAML root document and provides
 * the interface for reading each KPF line (which shows up as a YAML map)
 * into the KPF generic parser's packet buffer.
 */

#ifndef KWIVER_VITAL_KPF_YAML_PARSER_H_
#define KWIVER_VITAL_KPF_YAML_PARSER_H_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <arrows/kpf/yaml/kpf_parser_base.h>

#include <yaml-cpp/yaml.h>

namespace kwiver {
namespace vital {
namespace kpf {

class KPF_YAML_EXPORT kpf_yaml_parser_t: public kpf_parser_base_t
{
public:
  explicit kpf_yaml_parser_t( std::istream& is );
  ~kpf_yaml_parser_t() {}

  virtual bool get_status() const;
  virtual bool parse_next_record( packet_buffer_t& pb );
  virtual schema_style get_current_record_schema() const;
  virtual bool eof() const;

private:
  YAML::Node root;
  YAML::const_iterator current_record;
  schema_style current_record_schema;
};

} // ...kpf
} // ...vital
} // ...kwiver


#endif
