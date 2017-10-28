#ifndef KWIVER_VITAL_KPF_CANONICAL_TYPES_H_
#define KWIVER_VITAL_KPF_CANONICAL_TYPES_H_

#include <vital/kpf/vital_kpf_export.h>
#include <string>
#include <vector>

namespace kwiver {
namespace vital {
namespace kpf {

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
} // ...kpf
} // ...vital
} // ...kwiver

#endif
