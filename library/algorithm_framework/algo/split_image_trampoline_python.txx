// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPLIT_IMAGE_TRAMPOLINE_TXX
#define SPLIT_IMAGE_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/split_image.h>

namespace kwiver::vital::python {

template< class split_image_base = kwiver::vital::algo::split_image >
class split_image_trampoline
    : public algorithm_trampoline< split_image_base >
{
  public:
    using algorithm_trampoline< split_image_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  std::vector<std::shared_ptr<kwiver::vital::image_container> >
  split(::kwiver::vital::image_container_sptr img) const override
  {
    PYBIND11_OVERLOAD_PURE(
      std::vector<std::shared_ptr<kwiver::vital::image_container> >,
      kwiver::vital::algo::split_image,
      split,
      img
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
