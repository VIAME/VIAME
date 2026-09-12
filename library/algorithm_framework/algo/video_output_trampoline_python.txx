// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIDEO_OUTPUT_TRAMPOLINE_TXX
#define VIDEO_OUTPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/video_output.h>

namespace kwiver::vital::python {

template< class video_output_base = kwiver::vital::algo::video_output >
class video_output_trampoline
    : public algorithm_trampoline< video_output_base >
{
  public:
    using algorithm_trampoline< video_output_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string video_name, ::kwiver::vital::video_settings const * settings) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_output,
      open,
      video_name, settings
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_output,
      close,
      
      );
  }

  bool
  good() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_output,
      good,
      
      );
  }

  void
  add_image(::kwiver::vital::image_container_sptr const & image, ::kwiver::vital::timestamp const & ts) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_output,
      add_image,
      image, ts
      );
  }

  void
  add_image(::kwiver::vital::video_raw_image const & image) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::video_output,
      add_image,
      image
      );
  }

  void
  add_metadata(::kwiver::vital::metadata const & md) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_output,
      add_metadata,
      md
      );
  }

  void
  add_metadata(::kwiver::vital::video_raw_metadata const & md) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::video_output,
      add_metadata,
      md
      );
  }

  void
  add_uninterpreted_data(::kwiver::vital::video_uninterpreted_data const & misc_data) override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::video_output,
      add_uninterpreted_data,
      misc_data
      );
  }

  kwiver::vital::video_settings_sptr
  implementation_settings() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::video_settings_sptr,
      kwiver::vital::algo::video_output,
      implementation_settings,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
