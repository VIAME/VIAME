// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef MERGE_DETECTIONS_TRAMPOLINE_TXX
#define MERGE_DETECTIONS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/merge_detections.h>

namespace viame::python {

template< class merge_detections_base = viame::algo::merge_detections >
class merge_detections_trampoline
    : public algorithm_trampoline< merge_detections_base >
{
  public:
    using algorithm_trampoline< merge_detections_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::detected_object_set_sptr
  merge(::std::vector<std::shared_ptr<viame::detected_object_set> > const & sets) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::detected_object_set_sptr,
      viame::algo::merge_detections,
      merge,
      sets
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
