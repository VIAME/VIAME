// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SEGMENT_VIA_POINTS_TRAMPOLINE_TXX
#define SEGMENT_VIA_POINTS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/segment_via_points.h>

namespace viame::python {

template< class segment_via_points_base = viame::algo::segment_via_points >
class segment_via_points_trampoline
    : public algorithm_trampoline< segment_via_points_base >
{
  public:
    using algorithm_trampoline< segment_via_points_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::detected_object_set_sptr
  segment(::viame::image_container_sptr image, ::std::vector<viame::point<2, double> > const & points, ::std::vector<int> const & point_labels) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::detected_object_set_sptr,
      viame::algo::segment_via_points,
      segment,
      image, points, point_labels
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
