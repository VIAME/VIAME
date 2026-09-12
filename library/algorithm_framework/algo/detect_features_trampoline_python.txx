// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DETECT_FEATURES_TRAMPOLINE_TXX
#define DETECT_FEATURES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/detect_features.h>

namespace kwiver::vital::python {

template< class detect_features_base = kwiver::vital::algo::detect_features >
class detect_features_trampoline
    : public algorithm_trampoline< detect_features_base >
{
  public:
    using algorithm_trampoline< detect_features_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::feature_set_sptr
  detect(::kwiver::vital::image_container_sptr image_data, ::kwiver::vital::image_container_sptr mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::feature_set_sptr,
      kwiver::vital::algo::detect_features,
      detect,
      image_data, mask
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
