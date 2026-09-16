// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef IMAGE_OBJECT_DETECTOR_TRAMPOLINE_TXX
#define IMAGE_OBJECT_DETECTOR_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/image_object_detector.h>

namespace viame::python {

template< class image_object_detector_base = viame::algo::image_object_detector >
class image_object_detector_trampoline
    : public algorithm_trampoline< image_object_detector_base >
{
  public:
    using algorithm_trampoline< image_object_detector_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::detected_object_set_sptr
  detect(::viame::image_container_sptr image_data) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::detected_object_set_sptr,
      viame::algo::image_object_detector,
      detect,
      image_data
      );
  }

  std::vector<std::shared_ptr<viame::detected_object_set> >
  batch_detect(::std::vector<std::shared_ptr<viame::image_container> > const & images) const override
  {
    PYBIND11_OVERLOAD(
      std::vector<std::shared_ptr<viame::detected_object_set> >,
      viame::algo::image_object_detector,
      batch_detect,
      images
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
