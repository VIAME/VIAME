// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX
#define READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/read_track_descriptor_set.h>

namespace viame::python {

template< class read_track_descriptor_set_base = viame::algo::read_track_descriptor_set >
class read_track_descriptor_set_trampoline
    : public algorithm_trampoline< read_track_descriptor_set_base >
{
  public:
    using algorithm_trampoline< read_track_descriptor_set_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::read_track_descriptor_set,
      open,
      filename
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::read_track_descriptor_set,
      close,
      
      );
  }

  bool
  read_set(::viame::track_descriptor_set_sptr & set) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::read_track_descriptor_set,
      read_set,
      set
      );
  }

  void
  new_stream() override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::read_track_descriptor_set,
      new_stream,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
