// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DETECTED_OBJECT_FILTER_TRAMPOLINE_TXX
#define DETECTED_OBJECT_FILTER_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/detected_object_filter.h>

namespace viame::python {

template< class detected_object_filter_base = viame::algo::detected_object_filter >
class detected_object_filter_trampoline
    : public algorithm_trampoline< detected_object_filter_base >
{
  public:
    using algorithm_trampoline< detected_object_filter_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::detected_object_set_sptr
  filter(::viame::detected_object_set_sptr const input_set) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::detected_object_set_sptr,
      viame::algo::detected_object_filter,
      filter,
      input_set
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
