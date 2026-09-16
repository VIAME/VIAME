// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DETECT_MOTION_TRAMPOLINE_TXX
#define DETECT_MOTION_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/detect_motion.h>

namespace viame::python {

template< class detect_motion_base = viame::algo::detect_motion >
class detect_motion_trampoline
    : public algorithm_trampoline< detect_motion_base >
{
  public:
    using algorithm_trampoline< detect_motion_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::image_container_sptr
  process_image(::viame::timestamp const & ts, ::viame::image_container_sptr const image, bool reset_model) override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::image_container_sptr,
      viame::algo::detect_motion,
      process_image,
      ts, image, reset_model
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
