// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef WARP_IMAGE_TRAMPOLINE_TXX
#define WARP_IMAGE_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/warp_image.h>

namespace viame::python {

template< class warp_image_base = viame::algo::warp_image >
class warp_image_trampoline
    : public algorithm_trampoline< warp_image_base >
{
  public:
    using algorithm_trampoline< warp_image_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::image_container_sptr
  warp(::viame::image_container_sptr src_image, ::viame::image_container_sptr dst_image, ::viame::homography_sptr homography, ::viame::image_container_sptr alpha_mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::image_container_sptr,
      viame::algo::warp_image,
      warp,
      src_image, dst_image, homography, alpha_mask
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
