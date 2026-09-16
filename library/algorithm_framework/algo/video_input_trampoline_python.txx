// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIDEO_INPUT_TRAMPOLINE_TXX
#define VIDEO_INPUT_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/video_input.h>

namespace viame::python {

template< class video_input_base = viame::algo::video_input >
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
      viame::algo::video_input,
      open,
      video_name
      );
  }

  void
  close() override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::video_input,
      close,
      
      );
  }

  bool
  end_of_video() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_input,
      end_of_video,
      
      );
  }

  bool
  good() const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_input,
      good,
      
      );
  }

  size_t
  num_frames() const override
  {
    PYBIND11_OVERLOAD_PURE(
      size_t,
      viame::algo::video_input,
      num_frames,
      
      );
  }

  bool
  next_frame(::viame::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_input,
      next_frame,
      timeout
      );
  }

  bool
  seek_frame(::viame::timestamp::frame_t frame_number, ::viame::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_input,
      seek_frame,
      frame_number, timeout
      );
  }

  bool
  seek_time(::viame::timestamp::time_t time_usec, ::viame::time_usec_t timeout) override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::video_input,
      seek_time,
      time_usec, timeout
      );
  }

  viame::timestamp
  frame_timestamp() const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::timestamp,
      viame::algo::video_input,
      frame_timestamp,
      
      );
  }

  viame::image_container_sptr
  frame_image() override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::image_container_sptr,
      viame::algo::video_input,
      frame_image,
      
      );
  }

  viame::video_raw_image_sptr
  raw_frame_image() override
  {
    PYBIND11_OVERLOAD(
      viame::video_raw_image_sptr,
      viame::algo::video_input,
      raw_frame_image,
      
      );
  }

  viame::metadata_vector
  frame_metadata() override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::metadata_vector,
      viame::algo::video_input,
      frame_metadata,
      
      );
  }

  viame::video_raw_metadata_sptr
  raw_frame_metadata() override
  {
    PYBIND11_OVERLOAD(
      viame::video_raw_metadata_sptr,
      viame::algo::video_input,
      raw_frame_metadata,
      
      );
  }

  double
  frame_rate() override
  {
    PYBIND11_OVERLOAD(
      double,
      viame::algo::video_input,
      frame_rate,
      
      );
  }

  viame::path_t
  filename() const override
  {
    PYBIND11_OVERLOAD(
      viame::path_t,
      viame::algo::video_input,
      filename,
      
      );
  }

  viame::video_settings_sptr
  implementation_settings() const override
  {
    PYBIND11_OVERLOAD(
      viame::video_settings_sptr,
      viame::algo::video_input,
      implementation_settings,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
