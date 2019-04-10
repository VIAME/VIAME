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
 * @brief Base class for KPF parsers for various formats.
 *
 * Concrete format-specific instances of this class can be passed to
 * the kpf_reader_t to loop over KPF input.
 *
 */

#ifndef KWIVER_VITAL_KPF_PARSER_BASE_
#define KWIVER_VITAL_KPF_PARSER_BASE_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <arrows/kpf/yaml/kpf_yaml_schemas.h>

namespace kwiver {
namespace vital {
namespace kpf {

class kpf_parser_base_t
{
public:
  kpf_parser_base_t() {}
  virtual ~kpf_parser_base_t() {}

  virtual bool get_status() const = 0;
  virtual bool parse_next_record( packet_buffer_t& ) = 0;
  virtual schema_style get_current_record_schema() const = 0;
  virtual bool eof() const = 0;
};


} // ...kpf
} // ...vital
} // ...kwiver

#endif
