// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef IMAGE_IO_TRAMPOLINE_TXX
#define IMAGE_IO_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/image_io.h>

namespace viame::python {

template< class image_io_base = viame::algo::image_io >
class image_io_trampoline
    : public algorithm_trampoline< image_io_base >
{
  public:
    using algorithm_trampoline< image_io_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::image_container_sptr
  load_(::std::string const & filename) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::image_container_sptr,
      viame::algo::image_io,
      load_,
      filename
      );
  }

  void
  save_(::std::string const & filename, ::viame::image_container_sptr data) const override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::image_io,
      save_,
      filename, data
      );
  }

  viame::metadata_sptr
  load_metadata_(::std::string const & filename) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::metadata_sptr,
      viame::algo::image_io,
      load_metadata_,
      filename
      );
  }

  bool
  skip_path_validation_() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::image_io,
      skip_path_validation_,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
