// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef WARP_IMAGE_TRAMPOLINE_TXX
#define WARP_IMAGE_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/warp_image.h>

namespace kwiver::vital::python {

template< class warp_image_base = kwiver::vital::algo::warp_image >
class warp_image_trampoline
    : public algorithm_trampoline< warp_image_base >
{
  public:
    using algorithm_trampoline< warp_image_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::image_container_sptr
  warp(::kwiver::vital::image_container_sptr src_image, ::kwiver::vital::image_container_sptr dst_image, ::kwiver::vital::homography_sptr homography, ::kwiver::vital::image_container_sptr alpha_mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::warp_image,
      warp,
      src_image, dst_image, homography, alpha_mask
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
