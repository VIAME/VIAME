#ifndef KWIVER_VITAL_KPF_CANONICAL_IO_H_
#define KWIVER_VITAL_KPF_CANONICAL_IO_H_

#include <vital/kpf/kpf_canonical_types.h>

namespace kwiver {
namespace vital {
namespace kpf {

struct kpf_io_adapter_base;

template< typename T >
struct writer
{};

template <>
struct VITAL_KPF_EXPORT writer< canonical::bbox_t >
{
  writer( const canonical::bbox_t& b, int d) : box(b), domain(d) {}
  const canonical::bbox_t& box;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT writer< canonical::poly_t >
{
  writer( const canonical::poly_t& p, int d) : poly(p), domain(d) {}
  const canonical::poly_t& poly;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT writer< canonical::id_t >
{
  writer( size_t i, int d ): id(i), domain(d) {}
  canonical::id_t id;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT writer< canonical::timestamp_t >
{
  writer( double t, int d ): ts(t), domain(d) {}
  canonical::timestamp_t ts;
  int domain;
};

template<>
struct VITAL_KPF_EXPORT writer< canonical::kv_t >
{
  writer( const std::string& k, const std::string& v ): kv(k,v) {}
  canonical::kv_t kv;
  // key/value has no domain
};

template<>
struct VITAL_KPF_EXPORT writer< canonical::conf_t >
{
  writer( double c, int d ): conf(c), domain(d) {}
  canonical::conf_t conf;
  int domain;
};

template<>
struct VITAL_KPF_EXPORT writer< canonical::meta_t >
{
  writer( const std::string& t ): meta(t) {}
  canonical::meta_t meta;
  int domain;
};

template< typename T >
struct reader
{};

template <>
struct VITAL_KPF_EXPORT reader< canonical::bbox_t >
{
  reader( kpf_io_adapter_base& b, int d): box_adapter(b), domain(d) {}
  kpf_io_adapter_base& box_adapter;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::poly_t >
{
  reader( kpf_io_adapter_base& b, int d): poly_adapter(b), domain(d) {}
  kpf_io_adapter_base& poly_adapter;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::id_t >
{
  reader( int& id, int d ): id_ref(id), domain(d) {}
  int& id_ref;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::timestamp_t >
{
private:
  int i_dummy;
  unsigned u_dummy;
  double d_dummy;

public:
  enum which_t {to_int, to_unsigned, to_double};
  reader( int& ts, int d ): which( to_int), int_ts(ts), unsigned_ts(u_dummy), double_ts( d_dummy ),  domain(d) {}
  reader( unsigned& ts, int d ): which( to_unsigned ), int_ts( i_dummy), unsigned_ts(ts), double_ts( d_dummy ), domain(d) {}
  reader( double& ts, int d ): which( to_double ), int_ts( i_dummy ), unsigned_ts( u_dummy ), double_ts(ts), domain(d) {}
  which_t which;
  int& int_ts;
  unsigned& unsigned_ts;
  double& double_ts;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::kv_t >
{
  reader( const std::string& k, std::string& v ): key(k), val(v) {}
  std::string key;
  std::string& val;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::conf_t >
{
  reader( double& c, int d): conf(c), domain(d) {}
  double& conf;
  int domain;
};

template <>
struct VITAL_KPF_EXPORT reader< canonical::meta_t >
{
  reader( std::string& t): txt(t) {}
  std::string& txt;
};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
