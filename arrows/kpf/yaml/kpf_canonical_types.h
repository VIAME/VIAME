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
 * \brief Canonical KPF types.
 */

#ifndef KWIVER_VITAL_KPF_CANONICAL_TYPES_H_
#define KWIVER_VITAL_KPF_CANONICAL_TYPES_H_

#include <arrows/kpf/yaml/kpf_yaml_export.h>

#include <string>
#include <vector>

namespace kwiver {
namespace vital {
namespace kpf {

namespace canonical
{

struct KPF_YAML_EXPORT bbox_t
{
  enum {IMAGE_COORDS = 0};
  double x1, y1, x2, y2;
  bbox_t( double a, double b, double c, double d): x1(a), y1(b), x2(c), y2(d) {}
};

struct KPF_YAML_EXPORT id_t
{
  enum {DETECTION_ID=0, TRACK_ID, EVENT_ID };
  size_t d;
  explicit id_t( size_t i ): d(i) {}
  id_t(): d(0) {}
};

struct KPF_YAML_EXPORT timestamp_t
{
  enum {FRAME_NUMBER=0 };
  double d;
  explicit timestamp_t( double ts ): d(ts) {}
};

struct KPF_YAML_EXPORT timestamp_range_t
{
  timestamp_range_t( double a, double b ): start(a), stop(b) {}
  timestamp_range_t(): start(0), stop(0) {}
  double start, stop;
};

struct KPF_YAML_EXPORT kv_t
{
  std::string key, val;
  kv_t( const std::string& k, const std::string& v ): key(k), val(v) {}
};

struct KPF_YAML_EXPORT conf_t
{
  double d;
  explicit conf_t( double conf ): d(conf) {}
};

struct KPF_YAML_EXPORT eval_t
{
  double d;
  explicit eval_t( double score ): d(score) {}
};

struct KPF_YAML_EXPORT poly_t
{
  enum {IMAGE_COORDS = 0};
  std::vector< std::pair< double, double > > xy;
  poly_t( const std::vector< std::pair< double, double > >& p ): xy(p) {}
  poly_t() {}
};

struct KPF_YAML_EXPORT meta_t
{
  std::string txt;
  meta_t( const std::string& t ): txt(t) {}
  meta_t() {}
};

struct KPF_YAML_EXPORT activity_t
{
  struct scoped_tsr_t
  {
    int domain;
    timestamp_range_t tsr;
  };
  struct actor_t
  {
    int id_domain;
    id_t id;
    std::vector< scoped_tsr_t > actor_timespan;
  };
  std::string activity_name;
  id_t activity_id;
  int activity_id_domain;
  std::vector< scoped_tsr_t > timespan;
  std::vector< actor_t > actors;
  std::vector< kv_t > attributes;
};

} // ...canonical
} // ...kpf
} // ...vital
} // ...kwiver

#endif
