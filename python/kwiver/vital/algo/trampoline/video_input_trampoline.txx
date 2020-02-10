/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


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

#endif
