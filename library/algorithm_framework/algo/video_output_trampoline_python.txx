// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIDEO_OUTPUT_TRAMPOLINE_TXX
#define VIDEO_OUTPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/video_output.h>

namespace viame::python {

template< class video_output_base = viame::algo::video_output >
class video_output_trampoline
    : public algorithm_trampoline< video_output_base >
{
  public:
    using algorithm_trampoline< video_output_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string video_name, ::viame::video_settings const * settings) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::video_output,
      open,
      video_name, settings
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::video_output,
      close,
      
      );
  }

  bool
  good() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_output,
      good,
      
      );
  }

  void
  add_image(::viame::image_container_sptr const & image, ::viame::timestamp const & ts) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::video_output,
      add_image,
      image, ts
      );
  }

  void
  add_image(::viame::video_raw_image const & image) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::video_output,
      add_image,
      image
      );
  }

  void
  add_metadata(::viame::metadata const & md) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::video_output,
      add_metadata,
      md
      );
  }

  void
  add_metadata(::viame::video_raw_metadata const & md) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::video_output,
      add_metadata,
      md
      );
  }

  void
  add_uninterpreted_data(::viame::video_uninterpreted_data const & misc_data) override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::video_output,
      add_uninterpreted_data,
      misc_data
      );
  }

  viame::video_settings_sptr
  implementation_settings() const override
  {
    PYBIND11_OVERLOAD(
      viame::video_settings_sptr,
      viame::algo::video_output,
      implementation_settings,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
