// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef MATCH_FEATURES_TRAMPOLINE_TXX
#define MATCH_FEATURES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/match_features.h>

namespace viame::python {

template< class match_features_base = viame::algo::match_features >
class match_features_trampoline
    : public algorithm_trampoline< match_features_base >
{
  public:
    using algorithm_trampoline< match_features_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::match_set_sptr
  match(::viame::feature_set_sptr feat1, ::viame::descriptor_set_sptr desc1, ::viame::feature_set_sptr feat2, ::viame::descriptor_set_sptr desc2) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::match_set_sptr,
      viame::algo::match_features,
      match,
      feat1, desc1, feat2, desc2
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
