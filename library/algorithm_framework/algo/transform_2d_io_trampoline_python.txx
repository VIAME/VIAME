// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRANSFORM_2D_IO_TRAMPOLINE_TXX
#define TRANSFORM_2D_IO_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/transform_2d_io.h>

namespace kwiver::vital::python {

template< class transform_2d_io_base = kwiver::vital::algo::transform_2d_io >
class transform_2d_io_trampoline
    : public algorithm_trampoline< transform_2d_io_base >
{
  public:
    using algorithm_trampoline< transform_2d_io_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::transform_2d_sptr
  load_(::std::string const & filename) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::transform_2d_sptr,
      kwiver::vital::algo::transform_2d_io,
      load_,
      filename
      );
  }

  void
  save_(::std::string const & filename, ::kwiver::vital::transform_2d_sptr data) const override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::transform_2d_io,
      save_,
      filename, data
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
