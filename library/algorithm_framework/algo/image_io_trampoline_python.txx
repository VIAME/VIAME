// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef IMAGE_IO_TRAMPOLINE_TXX
#define IMAGE_IO_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/image_io.h>

namespace kwiver::vital::python {

template< class image_io_base = kwiver::vital::algo::image_io >
class image_io_trampoline
    : public algorithm_trampoline< image_io_base >
{
  public:
    using algorithm_trampoline< image_io_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::image_container_sptr
  load_(::std::string const & filename) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::image_io,
      load_,
      filename
      );
  }

  void
  save_(::std::string const & filename, ::kwiver::vital::image_container_sptr data) const override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::image_io,
      save_,
      filename, data
      );
  }

  kwiver::vital::metadata_sptr
  load_metadata_(::std::string const & filename) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::metadata_sptr,
      kwiver::vital::algo::image_io,
      load_metadata_,
      filename
      );
  }

  bool
  skip_path_validation_() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::image_io,
      skip_path_validation_,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
