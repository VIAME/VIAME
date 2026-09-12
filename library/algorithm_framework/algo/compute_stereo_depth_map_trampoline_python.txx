// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef COMPUTE_STEREO_DEPTH_MAP_TRAMPOLINE_TXX
#define COMPUTE_STEREO_DEPTH_MAP_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/compute_stereo_depth_map.h>

namespace kwiver::vital::python {

template< class compute_stereo_depth_map_base = kwiver::vital::algo::compute_stereo_depth_map >
class compute_stereo_depth_map_trampoline
    : public algorithm_trampoline< compute_stereo_depth_map_base >
{
  public:
    using algorithm_trampoline< compute_stereo_depth_map_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::image_container_sptr
  compute(::kwiver::vital::image_container_sptr left_image, ::kwiver::vital::image_container_sptr right_image) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::compute_stereo_depth_map,
      compute,
      left_image, right_image
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
