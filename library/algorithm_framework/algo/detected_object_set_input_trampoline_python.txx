// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DETECTED_OBJECT_SET_INPUT_TRAMPOLINE_TXX
#define DETECTED_OBJECT_SET_INPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/detected_object_set_input.h>

namespace kwiver::vital::python {

template< class detected_object_set_input_base = kwiver::vital::algo::detected_object_set_input >
class detected_object_set_input_trampoline
    : public algorithm_trampoline< detected_object_set_input_base >
{
  public:
    using algorithm_trampoline< detected_object_set_input_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  new_stream() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::detected_object_set_input,
      new_stream,
      
      );
  }

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::detected_object_set_input,
      open,
      filename
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::detected_object_set_input,
      close,
      
      );
  }

  bool
  read_set(::kwiver::vital::detected_object_set_sptr & set, ::std::string & image_name) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::detected_object_set_input,
      read_set,
      set, image_name
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
