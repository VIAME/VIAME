// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DRAW_DETECTED_OBJECT_SET_TRAMPOLINE_TXX
#define DRAW_DETECTED_OBJECT_SET_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/draw_detected_object_set.h>

namespace viame::python {

template< class draw_detected_object_set_base = viame::algo::draw_detected_object_set >
class draw_detected_object_set_trampoline
    : public algorithm_trampoline< draw_detected_object_set_base >
{
  public:
    using algorithm_trampoline< draw_detected_object_set_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::image_container_sptr
  draw(::viame::detected_object_set_sptr detected_set, ::viame::image_container_sptr image) override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::image_container_sptr,
      viame::algo::draw_detected_object_set,
      draw,
      detected_set, image
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
