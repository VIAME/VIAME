// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef FILTER_FEATURES_TRAMPOLINE_TXX
#define FILTER_FEATURES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/filter_features.h>

namespace kwiver::vital::python {

template< class filter_features_base = kwiver::vital::algo::filter_features >
class filter_features_trampoline
    : public algorithm_trampoline< filter_features_base >
{
  public:
    using algorithm_trampoline< filter_features_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::feature_set_sptr
  filter(::kwiver::vital::feature_set_sptr feat, ::std::vector<unsigned long> & indices) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::feature_set_sptr,
      kwiver::vital::algo::filter_features,
      filter,
      feat, indices
      );
  }

  kwiver::vital::feature_set_sptr
  filter(::kwiver::vital::feature_set_sptr input) const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::feature_set_sptr,
      kwiver::vital::algo::filter_features,
      filter,
      input
      );
  }

  kwiver::vital::algo::filter_features::filter_return_value
  filter(::kwiver::vital::feature_set_sptr feat, ::kwiver::vital::descriptor_set_sptr descr) const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::algo::filter_features::filter_return_value,
      kwiver::vital::algo::filter_features,
      filter,
      feat, descr
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
