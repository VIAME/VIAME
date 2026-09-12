// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef IMAGE_FILTER_TRAMPOLINE_TXX
#define IMAGE_FILTER_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/image_filter.h>

namespace kwiver::vital::python {

template< class image_filter_base = kwiver::vital::algo::image_filter >
class image_filter_trampoline
    : public algorithm_trampoline< image_filter_base >
{
  public:
    using algorithm_trampoline< image_filter_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::image_container_sptr
  filter(::kwiver::vital::image_container_sptr image_data) override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::image_filter,
      filter,
      image_data
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
