// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef REFINE_DETECTIONS_TRAMPOLINE_TXX
#define REFINE_DETECTIONS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/refine_detections.h>

namespace viame::python {

template< class refine_detections_base = viame::algo::refine_detections >
class refine_detections_trampoline
    : public algorithm_trampoline< refine_detections_base >
{
  public:
    using algorithm_trampoline< refine_detections_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::detected_object_set_sptr
  refine(::viame::image_container_sptr image_data, ::viame::detected_object_set_sptr detections) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::detected_object_set_sptr,
      viame::algo::refine_detections,
      refine,
      image_data, detections
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
