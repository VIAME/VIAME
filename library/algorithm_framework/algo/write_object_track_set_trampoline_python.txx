// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef WRITE_OBJECT_TRACK_SET_TRAMPOLINE_TXX
#define WRITE_OBJECT_TRACK_SET_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/write_object_track_set.h>

namespace kwiver::vital::python {

template< class write_object_track_set_base = kwiver::vital::algo::write_object_track_set >
class write_object_track_set_trampoline
    : public algorithm_trampoline< write_object_track_set_base >
{
  public:
    using algorithm_trampoline< write_object_track_set_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string const & filename) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::write_object_track_set,
      open,
      filename
      );
  }

  void
  use_stream(::std::ostream * strm) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::write_object_track_set,
      use_stream,
      strm
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::write_object_track_set,
      close,
      
      );
  }

  void
  write_set(::kwiver::vital::object_track_set_sptr const & set, ::kwiver::vital::timestamp const & ts, ::std::string const & frame_identifier) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::write_object_track_set,
      write_set,
      set, ts, frame_identifier
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
