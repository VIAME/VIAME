// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIDEO_INPUT_TRAMPOLINE_TXX
#define VIDEO_INPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/video_input.h>

namespace kwiver::vital::python {

template< class video_input_base = kwiver::vital::algo::video_input >
class video_input_trampoline
    : public algorithm_trampoline< video_input_base >
{
  public:
    using algorithm_trampoline< video_input_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  open(::std::string video_name) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_input,
      open,
      video_name
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::video_input,
      close,
      
      );
  }

  bool
  end_of_video() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_input,
      end_of_video,
      
      );
  }

  bool
  good() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_input,
      good,
      
      );
  }

  size_t
  num_frames() const override
  {
    PYBIND11_OVERLOAD_PURE(
      size_t,
      kwiver::vital::algo::video_input,
      num_frames,
      
      );
  }

  bool
  next_frame(::kwiver::vital::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_input,
      next_frame,
      timeout
      );
  }

  bool
  seek_frame(::kwiver::vital::timestamp::frame_t frame_number, ::kwiver::vital::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_input,
      seek_frame,
      frame_number, timeout
      );
  }

  bool
  seek_time(::kwiver::vital::timestamp::time_t time_usec, ::kwiver::vital::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      kwiver::vital::algo::video_input,
      seek_time,
      time_usec, timeout
      );
  }

  kwiver::vital::timestamp
  frame_timestamp() const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::timestamp,
      kwiver::vital::algo::video_input,
      frame_timestamp,
      
      );
  }

  kwiver::vital::image_container_sptr
  frame_image() override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::image_container_sptr,
      kwiver::vital::algo::video_input,
      frame_image,
      
      );
  }

  kwiver::vital::video_raw_image_sptr
  raw_frame_image() override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::video_raw_image_sptr,
      kwiver::vital::algo::video_input,
      raw_frame_image,
      
      );
  }

  kwiver::vital::metadata_vector
  frame_metadata() override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::metadata_vector,
      kwiver::vital::algo::video_input,
      frame_metadata,
      
      );
  }

  kwiver::vital::video_raw_metadata_sptr
  raw_frame_metadata() override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::video_raw_metadata_sptr,
      kwiver::vital::algo::video_input,
      raw_frame_metadata,
      
      );
  }

  double
  frame_rate() override
  {
    PYBIND11_OVERLOAD(
      double,
      kwiver::vital::algo::video_input,
      frame_rate,
      
      );
  }

  kwiver::vital::path_t
  filename() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::path_t,
      kwiver::vital::algo::video_input,
      filename,
      
      );
  }

  kwiver::vital::video_settings_sptr
  implementation_settings() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::video_settings_sptr,
      kwiver::vital::algo::video_input,
      implementation_settings,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
