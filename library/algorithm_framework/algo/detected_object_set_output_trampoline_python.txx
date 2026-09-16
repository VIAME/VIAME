// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX
#define DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/detected_object_set_output.h>

namespace viame::python {

template< class detected_object_set_output_base = viame::algo::detected_object_set_output >
class detected_object_set_output_trampoline
    : public algorithm_trampoline< detected_object_set_output_base >
{
  public:
    using algorithm_trampoline< detected_object_set_output_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::detected_object_set_output,
      open,
      filename
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::detected_object_set_output,
      close,
      
      );
  }

  void
  write_set(::viame::detected_object_set_sptr const set, ::std::string const & image_path) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::detected_object_set_output,
      write_set,
      set, image_path
      );
  }

  void
  complete() override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::detected_object_set_output,
      complete,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
