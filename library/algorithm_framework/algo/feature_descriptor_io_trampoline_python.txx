// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef FEATURE_DESCRIPTOR_IO_TRAMPOLINE_TXX
#define FEATURE_DESCRIPTOR_IO_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>

namespace viame::python {

template< class feature_descriptor_io_base = viame::algo::feature_descriptor_io >
class feature_descriptor_io_trampoline
    : public algorithm_trampoline< feature_descriptor_io_base >
{
  public:
    using algorithm_trampoline< feature_descriptor_io_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  load_(::std::string const & filename, ::viame::feature_set_sptr & feat, ::viame::descriptor_set_sptr & desc) const override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::feature_descriptor_io,
      load_,
      filename, feat, desc
      );
  }

  void
  save_(::std::string const & filename, ::viame::feature_set_sptr feat, ::viame::descriptor_set_sptr desc) const override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::feature_descriptor_io,
      save_,
      filename, feat, desc
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
