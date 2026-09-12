// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef WRITE_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX
#define WRITE_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/write_track_descriptor_set.h>

namespace kwiver::vital::python {

template< class write_track_descriptor_set_base = kwiver::vital::algo::write_track_descriptor_set >
class write_track_descriptor_set_trampoline
    : public algorithm_trampoline< write_track_descriptor_set_base >
{
  public:
    using algorithm_trampoline< write_track_descriptor_set_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::write_track_descriptor_set,
      open,
      filename
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::write_track_descriptor_set,
      close,
      
      );
  }

  void
  write_set(::kwiver::vital::track_descriptor_set_sptr const set, ::std::string const & source_id) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::write_track_descriptor_set,
      write_set,
      set, source_id
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
