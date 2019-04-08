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
 * \brief Readers and writers for canonical types.
 *
 * These classes have op>> and op<< defined in conjunction with the
 * kpf_reader to allow (hopefully) intuitive reading and writing of KPF.
 *
 * The design here probably needs to be rethought. The canonical implementations
 * are broken out, so that the user doesn't have to convert on their end, but
 * it's a little awkward internally.
 *
 */

#ifndef KWIVER_VITAL_KPF_CANONICAL_IO_H_
#define KWIVER_VITAL_KPF_CANONICAL_IO_H_

#include <arrows/kpf/yaml/kpf_canonical_types.h>
#include <arrows/kpf/yaml/kpf_canonical_io_adapter_base.h>

namespace kwiver {
namespace vital {
namespace kpf {


/**
 * \file
 * \brief Writers.
 *
 */

template< typename T >
struct writer
{};

template <>
struct KPF_YAML_EXPORT writer< canonical::bbox_t >
{
  writer( const canonical::bbox_t& b, int d) : box(b), domain(d) {}
  explicit writer( const canonical::scoped< canonical::bbox_t >& s )
    : box( s.t ), domain( s.domain ) {}
  const canonical::bbox_t& box;
  int domain;
};

template <>
struct KPF_YAML_EXPORT writer< canonical::poly_t >
{
  writer( const canonical::poly_t& p, int d) : poly(p), domain(d) {}
  explicit writer( const canonical::scoped< canonical::poly_t >& s )
    : poly( s.t ), domain( s.domain ) {}
  const canonical::poly_t& poly;
  int domain;
};

template <>
struct KPF_YAML_EXPORT writer< canonical::activity_t >
{
  writer( const canonical::activity_t& a, int d) : activity(a), domain(d) {}
  explicit writer( const canonical::scoped< canonical::activity_t >& s )
          : activity( s.t ), domain( s.domain ) {}
  const canonical::activity_t& activity;
  int domain;
};

template <>
struct KPF_YAML_EXPORT writer< canonical::id_t >
{
  writer( size_t i, int d ): id(i), domain(d) {}
  writer( const canonical::id_t& i, int d )
    : id( i ), domain( d ) {}
  explicit writer( const canonical::scoped< canonical::id_t >& s )
    : id( s.t ), domain( s.domain ) {}
  canonical::id_t id;
  int domain;
};

template <>
struct KPF_YAML_EXPORT writer< canonical::timestamp_t >
{
  writer( double t, int d ): ts(t), domain(d) {}
  writer( const canonical::timestamp_t& t, int d )
    : ts( t ), domain( d ) {}
  explicit writer( const canonical::scoped< canonical::timestamp_t >& s )
    : ts( s.t ), domain( s.domain ) {}
  canonical::timestamp_t ts;
  int domain;
};

template<>
struct KPF_YAML_EXPORT writer< canonical::kv_t >
{
  writer( const std::string& k, const std::string& v ): kv(k,v) {}
  explicit writer( const canonical::kv_t& kv_in ): kv( kv_in ) {}
  canonical::kv_t kv;
  // key/value has no domain, thus no scoped version
};

template<>
struct KPF_YAML_EXPORT writer< canonical::conf_t >
{
  writer( double c, int d ): conf(c), domain(d) {}
  writer( const canonical::conf_t& c, int d )
    : conf( c ), domain( d ) {}
  explicit writer( const canonical::scoped< canonical::conf_t >& s )
    : conf( s.t ), domain( s.domain ) {}
  canonical::conf_t conf;
  int domain;
};

template<>
struct KPF_YAML_EXPORT writer< canonical::cset_t >
{
  writer( const canonical::cset_t& c, int d ): cset(c), domain(d) {}
  explicit writer( const canonical::scoped< canonical::cset_t >& s )
    : cset( s.t ), domain( s.domain ) {}
  canonical::cset_t cset;
  int domain;
};

template<>
struct KPF_YAML_EXPORT writer< canonical::eval_t >
{
  writer( double c, int d ): eval(c), domain(d) {}
  writer( const canonical::eval_t& e, int d )
    : eval( e ), domain( d ) {}
  explicit writer( const canonical::scoped< canonical::eval_t >& s )
    : eval( s.t ), domain( s.domain ) {}
  canonical::eval_t eval;
  int domain;
};

template<>
struct KPF_YAML_EXPORT writer< canonical::meta_t >
{
  writer( const std::string& t ): meta(t) {}
  canonical::meta_t meta;
  // meta has no domain, thus no scoped version
};

template<>
struct KPF_YAML_EXPORT writer< canonical::timestamp_range_t >
{
  writer( double start, double stop ): tsr(start, stop) {}
  writer( const canonical::timestamp_range_t& t, int d )
    : tsr( t ), domain( d ) {}
  explicit writer( const canonical::scoped< canonical::timestamp_range_t >& s )
    : tsr( s.t ), domain( s.domain ) {}
  canonical::timestamp_range_t tsr;
  int domain;
};

/**
 * \file
 * \brief Readers.
 *
 */

template< typename T >
struct reader
{};

template <>
struct KPF_YAML_EXPORT reader< canonical::bbox_t >
{
  reader( kpf_canonical_io_adapter_base& b, int d): box_adapter(b), domain(d) {}
  kpf_canonical_io_adapter_base& box_adapter;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::poly_t >
{
  reader( kpf_canonical_io_adapter_base& b, int d): poly_adapter(b), domain(d) {}
  kpf_canonical_io_adapter_base& poly_adapter;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::activity_t >
{
  reader( kpf_canonical_io_adapter_base& b, int d): act_adapter(b), domain(d) {}
  kpf_canonical_io_adapter_base& act_adapter;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::id_t >
{
  reader(size_t& id, int d ): id_ref(id), domain(d) {}
  size_t& id_ref;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::timestamp_t >
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
struct KPF_YAML_EXPORT reader< canonical::kv_t >
{
  reader( const std::string& k, std::string& v ): key(k), val(v) {}
  std::string key;
  std::string& val;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::conf_t >
{
  reader( double& c, int d): conf(c), domain(d) {}
  double& conf;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::cset_t >
{
  reader( canonical::cset_t& c, int d): cset(c), domain(d) {}
  canonical::cset_t& cset;
  int domain;
};


template <>
struct KPF_YAML_EXPORT reader< canonical::eval_t >
{
  reader( double& s, int d): score(s), domain(d) {}
  double& score;
  int domain;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::meta_t >
{
  reader( std::string& t): txt(t) {}
  std::string& txt;
};

template <>
struct KPF_YAML_EXPORT reader< canonical::timestamp_range_t >
{
private:
  std::pair<int, int> i_dummy;
  std::pair<unsigned, unsigned> u_dummy;
  std::pair<double, double> d_dummy;

public:
  enum which_t {to_int, to_unsigned, to_double};
  reader( std::pair<int, int>& ts, int d ): which( to_int), int_ts(ts), unsigned_ts(u_dummy), double_ts( d_dummy ),  domain(d) {}
  reader( std::pair<unsigned, unsigned>& ts, int d ): which( to_unsigned ), int_ts( i_dummy), unsigned_ts(ts), double_ts( d_dummy ), domain(d) {}
  reader( std::pair<double, double>& ts, int d ): which( to_double ), int_ts( i_dummy ), unsigned_ts( u_dummy ), double_ts(ts), domain(d) {}
  which_t which;
  std::pair<int, int>& int_ts;
  std::pair<unsigned, unsigned>& unsigned_ts;
  std::pair<double, double>& double_ts;
  int domain;
};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
