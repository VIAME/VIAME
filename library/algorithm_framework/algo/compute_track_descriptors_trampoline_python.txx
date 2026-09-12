// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef COMPUTE_TRACK_DESCRIPTORS_TRAMPOLINE_TXX
#define COMPUTE_TRACK_DESCRIPTORS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/compute_track_descriptors.h>

namespace kwiver::vital::python {

template< class compute_track_descriptors_base = kwiver::vital::algo::compute_track_descriptors >
class compute_track_descriptors_trampoline
    : public algorithm_trampoline< compute_track_descriptors_base >
{
  public:
    using algorithm_trampoline< compute_track_descriptors_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::track_descriptor_set_sptr
  compute(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image_data, ::kwiver::vital::object_track_set_sptr tracks) override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::track_descriptor_set_sptr,
      kwiver::vital::algo::compute_track_descriptors,
      compute,
      ts, image_data, tracks
      );
  }

  kwiver::vital::track_descriptor_set_sptr
  flush() override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::track_descriptor_set_sptr,
      kwiver::vital::algo::compute_track_descriptors,
      flush,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
