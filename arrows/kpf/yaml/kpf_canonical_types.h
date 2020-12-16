// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Canonical KPF types.
 */

#ifndef KWIVER_VITAL_KPF_CANONICAL_TYPES_H_
#define KWIVER_VITAL_KPF_CANONICAL_TYPES_H_

#include <arrows/kpf/yaml/kpf_yaml_export.h>

#include <arrows/kpf/yaml/kpf_packet_header.h>

#include <string>
#include <vector>
#include <map>

namespace kwiver {
namespace vital {
namespace kpf {

namespace canonical
{

//
// separate the *types* from the *domains*
//

template< typename T >
struct scoped
{
  T t;
  int domain;
  scoped(): t( T() ), domain( packet_header_t::NO_DOMAIN ) {}
  explicit scoped( const T& t_in ): t(t_in), domain( packet_header_t::NO_DOMAIN ) {}
  scoped( const T& t_in, int domain_in ): t(t_in), domain(domain_in) {}
};

struct KPF_YAML_EXPORT bbox_t
{
  enum {IMAGE_COORDS = 0};
  double x1, y1, x2, y2;
  bbox_t(): x1(0), y1(0), x2(0), y2(0) {}
  bbox_t( double a, double b, double c, double d): x1(a), y1(b), x2(c), y2(d) {}
};

struct KPF_YAML_EXPORT id_t
{
  enum {DETECTION_ID=0, TRACK_ID, EVENT_ID };
  uint64_t d;
  explicit id_t( uint64_t i ): d(i) {}
  id_t(): d(0) {}
};

struct KPF_YAML_EXPORT timestamp_t
{
  enum {FRAME_NUMBER=0 };
  double d;
  timestamp_t(): d(0.0) {}
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
  kv_t(): key(""), val("") {}
  kv_t( const std::string& k, const std::string& v );
};

struct KPF_YAML_EXPORT conf_t
{
  double d;
  conf_t(): d(0.0) {}
  explicit conf_t( double conf ): d(conf) {}
};

struct KPF_YAML_EXPORT cset_t
{
  std::map< std::string, double > d;
  cset_t() {}
  cset_t( const std::map< std::string, double >& in_d ): d( in_d ) {}
  cset_t( const cset_t& other ): d( other.d ) {}
};

struct KPF_YAML_EXPORT eval_t
{
  double d;
  eval_t(): d(0.0) {}
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
  meta_t():txt("uninitialized-meta") {}
};

struct KPF_YAML_EXPORT activity_t
{
  struct actor_t
  {
    scoped< id_t > actor_id;
    std::vector< scoped< timestamp_range_t > > actor_timespan;
  };

  cset_t activity_labels;
  scoped< id_t > activity_id;
  std::vector< scoped< timestamp_range_t > > timespan;
  std::vector< actor_t > actors;
  std::vector< kv_t > attributes;
  std::vector< scoped< eval_t > > evals;

  activity_t():
    activity_id( id_t(), -1 )
    {}

};

} // ...canonical
} // ...kpf
} // ...vital
} // ...kwiver

#endif
