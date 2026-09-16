// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRAIN_DETECTOR_TRAMPOLINE_TXX
#define TRAIN_DETECTOR_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/train_detector.h>

namespace viame::python {

template< class train_detector_base = viame::algo::train_detector >
class train_detector_trampoline
    : public algorithm_trampoline< train_detector_base >
{
  public:
    using algorithm_trampoline< train_detector_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  add_data_from_disk(::viame::category_hierarchy_sptr object_labels, ::std::vector<std::basic_string<char> > train_image_names, ::std::vector<std::shared_ptr<viame::detected_object_set> > train_groundtruth, ::std::vector<std::basic_string<char> > test_image_names, ::std::vector<std::shared_ptr<viame::detected_object_set> > test_groundtruth) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::train_detector,
      add_data_from_disk,
      object_labels, train_image_names, train_groundtruth, test_image_names, test_groundtruth
      );
  }

  void
  add_data_from_memory(::viame::category_hierarchy_sptr object_labels, ::std::vector<std::shared_ptr<viame::image_container> > train_images, ::std::vector<std::shared_ptr<viame::detected_object_set> > train_groundtruth, ::std::vector<std::shared_ptr<viame::image_container> > test_images, ::std::vector<std::shared_ptr<viame::detected_object_set> > test_groundtruth) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::train_detector,
      add_data_from_memory,
      object_labels, train_images, train_groundtruth, test_images, test_groundtruth
      );
  }

  std::map<std::basic_string<char>, std::basic_string<char> >
  update_model() override
  {
    using update_model_return_t = std::map<std::basic_string<char>, std::basic_string<char> >;
    PYBIND11_OVERLOAD_PURE(
      update_model_return_t,
      viame::algo::train_detector,
      update_model,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
