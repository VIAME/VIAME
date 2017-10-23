#ifndef KWIVER_VITAL_KPF_PACKET_H_
#define KWIVER_VITAL_KPF_PACKET_H_

#include <vital/kpf/vital_kpf_export.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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

VITAL_KPF_EXPORT auto packet_header_cmp = []( const packet_header_t& lhs, const packet_header_t& rhs )
{ return ( lhs.style == rhs.style )
  ? (lhs.domain < rhs.domain)
  : (lhs.style < rhs.style);
};

namespace canonical
{

struct VITAL_KPF_EXPORT bbox_t
{
  enum {IMAGE_COORDS = 0};
  double x1, y1, x2, y2;
  bbox_t( double a, double b, double c, double d): x1(a), y1(b), x2(c), y2(d) {}
};

struct VITAL_KPF_EXPORT id_t
{
  enum {DETECTION_ID=0, TRACK_ID, EVENT_ID };
  size_t d;
  explicit id_t( size_t i ): d(i) {}
};

struct VITAL_KPF_EXPORT timestamp_t
{
  enum {FRAME_NUMBER=0 };
  double d;
  explicit timestamp_t( double ts ): d(ts) {}
};

struct VITAL_KPF_EXPORT timestamp_range_t
{
  timestamp_range_t( double a, double b ): start(a), stop(b) {}
  double start, stop;
};

struct VITAL_KPF_EXPORT kv_t
{
  std::string key, val;
  kv_t( const std::string& k, const std::string& v ): key(k), val(v) {}
};

struct VITAL_KPF_EXPORT conf_t
{
  double d;
  explicit conf_t( double conf ): d(conf) {}
};

struct VITAL_KPF_EXPORT poly_t
{
  enum {IMAGE_COORDS = 0};
  std::vector< std::pair< double, double > > xy;
  poly_t( const std::vector< std::pair< double, double > >& p ): xy(p) {}
  poly_t() {}
};

struct VITAL_KPF_EXPORT meta_t
{
  std::string txt;
  meta_t( const std::string& t ): txt(t) {}
  meta_t() {}
};

} // ...canonical

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
  };
  packet_t(): header( packet_header_t() ) {}
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
