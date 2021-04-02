// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file video_input_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<video_input> and video_input
 */

#ifndef VIDEO_INPUT_TRAMPOLINE_TXX
#define VIDEO_INPUT_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/video_input.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_vi_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::video_input > >
class algorithm_def_vi_trampoline :
      public algorithm_trampoline<algorithm_def_vi_base>
{
  public:
    using algorithm_trampoline<algorithm_def_vi_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::video_input>,
        type_name,
      );
    }
};

template< class video_input_base=
                kwiver::vital::algo::video_input >
class video_input_trampoline :
      public algorithm_def_vi_trampoline< video_input_base >
{
  public:
    using algorithm_def_vi_trampoline< video_input_base >::
              algorithm_def_vi_trampoline;

    void
    open( std::string video_name ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::video_input,
        open,
        video_name
      );
    }

    void
    close() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::video_input,
        close,
      );
    }

    bool
    end_of_video() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::video_input,
        end_of_video,
      );
    }

    bool
    good() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::video_input,
        good,
      );
    }

    bool
    seekable() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::video_input,
        seekable,
      );
    }

    size_t
    num_frames() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        size_t,
        kwiver::vital::algo::video_input,
        num_frames,
      );
    }

    bool
    next_frame( kwiver::vital::timestamp& ts,
                uint32_t timeout = 0) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::video_input,
        next_frame,
        ts,
        timeout
      );
    }

    bool
    seek_frame(kwiver::vital::timestamp& ts,
               kwiver::vital::timestamp::frame_t frame_number,
               uint32_t timeout = 0) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::video_input,
        seek_frame,
        ts,
        frame_number,
        timeout
      );
    }

    kwiver::vital::timestamp
    frame_timestamp() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::timestamp,
        kwiver::vital::algo::video_input,
        frame_timestamp,
      );
    }

    kwiver::vital::image_container_sptr
    frame_image() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::video_input,
        frame_image,
      );
    }

    kwiver::vital::metadata_vector
    frame_metadata() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::metadata_vector,
        kwiver::vital::algo::video_input,
        frame_metadata,
      );
    }

    kwiver::vital::metadata_map_sptr
    metadata_map() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::metadata_map_sptr,
        kwiver::vital::algo::video_input,
        metadata_map,
      );
    }

    double
    frame_rate() override
    {
      VITAL_PYBIND11_OVERLOAD(
        double,
        kwiver::vital::algo::video_input,
        frame_rate,
      );
    }
};

}
}
}

#endif
