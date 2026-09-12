// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef READ_OBJECT_TRACK_SET_TRAMPOLINE_TXX
#define READ_OBJECT_TRACK_SET_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/read_object_track_set.h>

namespace kwiver::vital::python {

template< class read_object_track_set_base = kwiver::vital::algo::read_object_track_set >
class read_object_track_set_trampoline
    : public algorithm_trampoline< read_object_track_set_base >
{
  public:
    using algorithm_trampoline< read_object_track_set_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::read_object_track_set,
      open,
      filename
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::read_object_track_set,
      close,
      
      );
  }

  bool
  read_set(::kwiver::vital::object_track_set_sptr & set) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::read_object_track_set,
      read_set,
      set
      );
  }

  void
  new_stream() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::read_object_track_set,
      new_stream,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
