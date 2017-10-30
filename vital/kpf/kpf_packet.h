#ifndef KWIVER_VITAL_KPF_PACKET_H_
#define KWIVER_VITAL_KPF_PACKET_H_

#include <vital/kpf/vital_kpf_export.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <vital/kpf/kpf_canonical_types.h>

namespace kwiver {
namespace vital {
namespace kpf {

enum class VITAL_KPF_EXPORT packet_style
{
  INVALID,  // invalid, uninitialized
  META,     // an uninterpreted string (consumes all following tokens)
  ID,       // a numeric identifier (detection, track, event ID)
  TS,       // timestamp
  TSR,      // timestamp range
  LOC,      // location (2d / 3d)
  GEOM,     // bounding box
  POLY,     // polygon
  CONF,     // a confidence value
  ACT,      // an activity
  EVAL,     // an evaluation result
  ATTR,     // an attribute
  TAG,      // a tag
  KV        // a generic key/value pair
};

struct VITAL_KPF_EXPORT packet_header_t
{
  enum { NO_DOMAIN = -1 };

  packet_style style;
  int domain;
  packet_header_t(): style( packet_style::INVALID ), domain( NO_DOMAIN ) {}
  packet_header_t( packet_style s, int d ): style(s), domain(d) {}
  packet_header_t( packet_style s ): style(s), domain( NO_DOMAIN ) {}
};

class VITAL_KPF_EXPORT packet_header_cmp
{
public:
  bool operator()( const packet_header_t& lhs, const packet_header_t& rhs ) const;
};

struct VITAL_KPF_EXPORT packet_t
{
  packet_header_t header;
  union
  {
    canonical::id_t id;
    canonical::timestamp_t timestamp;
    canonical::timestamp_range_t timestamp_range;
    canonical::bbox_t bbox;
    canonical::kv_t kv;
    canonical::conf_t conf;
    canonical::poly_t poly;
    canonical::meta_t meta;
    canonical::activity_t activity;
  };
  packet_t(): header( packet_header_t() ) {}
  packet_t( const packet_header_t& h ): header(h) {}
  ~packet_t();
  packet_t( const packet_t& other );
  packet_t& operator=( const packet_t& other );
};

std::ostream& VITAL_KPF_EXPORT operator<<( std::ostream& os, const packet_header_t& p );
std::ostream& VITAL_KPF_EXPORT operator<<( std::ostream& os, const packet_t& p );

packet_style VITAL_KPF_EXPORT str2style( const std::string& s );
std::string VITAL_KPF_EXPORT style2str( packet_style );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
