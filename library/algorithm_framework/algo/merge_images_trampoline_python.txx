// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef MERGE_IMAGES_TRAMPOLINE_TXX
#define MERGE_IMAGES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/merge_images.h>

namespace kwiver::vital::python {

template< class merge_images_base = kwiver::vital::algo::merge_images >
class merge_images_trampoline
    : public algorithm_trampoline< merge_images_base >
{
  public:
    using algorithm_trampoline< merge_images_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::image_container_sptr
  merge(::kwiver::vital::image_container_sptr image1, ::kwiver::vital::image_container_sptr image2) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::merge_images,
      merge,
      image1, image2
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
