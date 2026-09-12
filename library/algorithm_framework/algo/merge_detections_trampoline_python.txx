// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef MERGE_DETECTIONS_TRAMPOLINE_TXX
#define MERGE_DETECTIONS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/merge_detections.h>

namespace kwiver::vital::python {

template< class merge_detections_base = kwiver::vital::algo::merge_detections >
class merge_detections_trampoline
    : public algorithm_trampoline< merge_detections_base >
{
  public:
    using algorithm_trampoline< merge_detections_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::detected_object_set_sptr
  merge(::std::vector<std::shared_ptr<kwiver::vital::detected_object_set> > const & sets) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::detected_object_set_sptr,
      kwiver::vital::algo::merge_detections,
      merge,
      sets
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
