/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \file
 * \brief Implementation file for video input using FFMPEG.
 */

#include "ffmpeg_init.h"
#include "ffmpeg_video_input.h"

#include <vital/types/timestamp.h>
#include <vital/exceptions/io.h>
#include <vital/exceptions/video.h>
#include <vital/util/tokenize.h>
#include <vital/types/image_container.h>

#include <kwiversys/SystemTools.hxx>

#include <mutex>
#include <memory>
#include <vector>
#include <sstream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace kwiver {
namespace arrows {
namespace ffmpeg {

// ------------------------------------------------------------------
// Private implementation class
class ffmpeg_video_input::priv
{
public:
  /// Constructor
  priv() :
    f_format_context(nullptr),
    f_video_index(-1),
    f_data_index(-1),
    f_video_encoding(nullptr),
    f_video_stream(nullptr),
    f_frame(nullptr),
    f_software_context(nullptr),
    f_start_time(-1),
    f_frame_number_offset(0),
    video_path("")
  {
    f_packet.data = nullptr;
  }

  // f_* variables are FFMPEG specific

  AVFormatContext* f_format_context;
  int f_video_index;
  int f_data_index;
  AVCodecContext* f_video_encoding;
  AVStream* f_video_stream;
  AVFrame* f_frame;
  AVPacket f_packet;
  SwsContext* f_software_context;


  // Start time of the stream, to offset the pts when computing the frame number.
  // (in stream time base)
  int64_t f_start_time;

  // Presentation timestamp (in stream time base)
  int64_t f_pts;

  // Some codec/file format combinations need a frame number offset.
  // These codecs have a delay between reading packets and generating frames.
  unsigned f_frame_number_offset;

  // Name of video we opened
  std::string video_path;

  static std::mutex open_mutex;

  // For logging in priv methods
  vital::logger_handle_t logger;

  // Current image frame.
  vital::image_memory_sptr current_image_memory;
  kwiver::vital::image_container_sptr current_image;

  enum pixel_format
  {
    PIXEL_FORMAT_UNKNOWN = -1,

    PIXEL_FORMAT_RGB_24,
    PIXEL_FORMAT_RGB_24P,
    PIXEL_FORMAT_BGR_24,
    PIXEL_FORMAT_RGBA_32,
    PIXEL_FORMAT_RGBA_32P,
    PIXEL_FORMAT_RGB_565,
    PIXEL_FORMAT_RGB_555,

    PIXEL_FORMAT_YUV_444P,
    PIXEL_FORMAT_YUV_422P,
    PIXEL_FORMAT_YUV_420P,
    PIXEL_FORMAT_YVU_420P,
    PIXEL_FORMAT_YUV_411P,
    PIXEL_FORMAT_YUV_410P,
    PIXEL_FORMAT_UYV_444,
    PIXEL_FORMAT_YUYV_422,
    PIXEL_FORMAT_UYVY_422,
    PIXEL_FORMAT_UYVY_411,

    PIXEL_FORMAT_MONO_1,
    PIXEL_FORMAT_MONO_8,
    PIXEL_FORMAT_MONO_16,
    PIXEL_FORMAT_MONO_F32,
    PIXEL_FORMAT_RGB_F32,
    PIXEL_FORMAT_RGB_F32P,

    // Add values here
    PIXEL_FORMAT_ENUM_END
  };

  // ==================================================================
  /*
  * @brief Convert a ffmpeg pixel type to kwiver::vital::image_pixel_traits
  *
  * @return \b A kwiver::vital::image_pixel_traits
  */

  // ==================================================================
  /*
  * @brief Whether the video was opened.
  *
  * @return \b true if video was opened.
  */
  bool is_opened()
  {
    return this->f_start_time != -1;
  }

  // ==================================================================
  /*
  * @brief Open the given video.
  *
  * @return \b true if video was opened.
  */
  bool open(std::string video_name)
  {
    // Open the file
    int err = avformat_open_input(&this->f_format_context, this->video_path.c_str(), NULL, NULL);
    if (err != 0)
    {
      LOG_ERROR(this->logger, "Error " << err << " trying to open " << video_name);
      return false;
    }


    // Get the stream information by reading a bit of the file
    if (avformat_find_stream_info(this->f_format_context, NULL) < 0)
    {
      return false;
    }

    // Find a video stream, and optionally a data stream.
    // Use the first ones we find.
    this->f_video_index = -1;
    this->f_data_index = -1;
    AVCodecContext* codec_context_origin = NULL;
    for (unsigned i = 0; i < this->f_format_context->nb_streams; ++i)
    {
      AVCodecContext *const enc = this->f_format_context->streams[i]->codec;
      if (enc->codec_type == AVMEDIA_TYPE_VIDEO && this->f_video_index < 0)
      {
        this->f_video_index = i;
        codec_context_origin = enc;
      }
      else if (enc->codec_type == AVMEDIA_TYPE_DATA && this->f_data_index < 0)
      {
        this->f_data_index = i;
      }
    }

    if (this->f_video_index < 0)
    {
      LOG_ERROR(this->logger, "Error: could not find a video stream in " << this->video_path);
      return false;
    }

    if (this->f_data_index < 0)
    {
      LOG_INFO(this->logger, "No data stream available, using AVMEDIA_TYPE_UNKNOWN stream instead");
      // Fallback for the DATA stream if incorrectly coded as UNKNOWN.
      for (unsigned i = 0; i < this->f_format_context->nb_streams; ++i)
      {
        AVCodecContext *enc = this->f_format_context->streams[i]->codec;
        if (enc->codec_type == AVMEDIA_TYPE_UNKNOWN)
        {
          this->f_data_index = i;
        }
      }
    }

    av_dump_format(this->f_format_context, 0, this->video_path.c_str(), 0);

    // Open the stream
    AVCodec* codec = avcodec_find_decoder(codec_context_origin->codec_id);
    if (!codec)
    {
      LOG_ERROR(this->logger,
        "Error: Codec " << codec_context_origin->codec_descriptor
        << " (" << codec_context_origin->codec_id << ") not found");
      return false;
    }

    // Copy context
    this->f_video_encoding = avcodec_alloc_context3(codec);
    if (avcodec_copy_context(this->f_video_encoding, codec_context_origin) != 0)
    {
      LOG_ERROR(this->logger, "Error: Could not copy codec " << this->f_video_encoding->codec_id);
      return false;
    }

    // Open codec
    if (avcodec_open2(this->f_video_encoding, codec, NULL) < 0)
    {
      LOG_ERROR(this->logger, "Error: Could not open codec " << this->f_video_encoding->codec_id);
      return false;
    }

    this->f_video_stream = this->f_format_context->streams[this->f_video_index];
    this->f_frame = av_frame_alloc();

    if (this->f_video_stream->start_time == int64_t(1) << 63)
    {
      this->f_start_time = 0;
    }
    else
    {
      this->f_start_time = this->f_video_stream->start_time;
    }

    // The MPEG 2 codec has a latency of 1 frame when encoded in an AVI
    // stream, so the pts of the last packet (stored in pts) is
    // actually the next frame's pts.
    if (this->f_video_stream->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO &&
      std::string("avi") == this->f_format_context->iformat->name)
    {
      this->f_frame_number_offset = 1;
    }

    // Not sure if this does anything, but no harm either
    av_init_packet(&this->f_packet);
    this->f_packet.data = nullptr;
    this->f_packet.size = 0;

    return true;
  }

  // ==================================================================
  /*
  * @brief Advance to the next frame (but don't acquire an image).
  *
  * @return \b true if video was valid and we found a frame.
  */
  virtual bool advance()
  {
    // Quick return if the file isn't open.
    if (!this->is_opened())
    {
      return false;
    }

    // \todo - num_frames not implemented
    //// See the comment in num_frames().  This is to make sure that once
    //// we start reading frames, we'll never try to march to the end to
    //// figure out how many frames there are.
    //if (is_->num_frames_ == -2) {
    //  is_->num_frames_ = -1;
    //}

    if (this->f_packet.data)
    {
      av_free_packet(&this->f_packet);  // free previous packet
    }

    int got_picture = 0;

    // \todo - metada not implemented yet
    // clear the metadata from the previous frame
    //this->metadata.clear();

    while (got_picture == 0 && av_read_frame(this->f_format_context, &this->f_packet) >= 0)
    {
      // Make sure that the packet is from the actual video stream.
      if (this->f_packet.stream_index == this->f_video_index)
      {
        int err = avcodec_decode_video2(this->f_video_encoding,
          this->f_frame, &got_picture,
          &this->f_packet);
        if (err == AVERROR_INVALIDDATA)
        {// Ignore the frame and move to the next
          av_free_packet(&this->f_packet);
          continue;
        }
        if (err < 0)
        {
          LOG_ERROR(this->logger, "vidl_ffmpeg_istream: Error decoding packet");
          av_free_packet(&this->f_packet);
          return false;
        }

        this->f_pts = av_frame_get_best_effort_timestamp(this->f_frame);
        if (this->f_pts == AV_NOPTS_VALUE)
        {
          this->f_pts = 0;
        }
      }
      // \todo - No metadata support yet
      //// grab the metadata from this packet if from the metadata stream
      //else if (this->packet.stream_index == this->data_index)
      //{
      //  is_->metadata_.insert(is_->metadata_.end(), is_->packet_.data,
      //    is_->packet_.data + is_->packet_.size);
      //}

      if (!got_picture)
      {
        av_free_packet(&this->f_packet);
      }
    }

    // From ffmpeg apiexample.c: some codecs, such as MPEG, transmit the
    // I and P frame with a latency of one frame. You must do the
    // following to have a chance to get the last frame of the video.
    if (!got_picture)
    {
      av_init_packet(&this->f_packet);
      this->f_packet.data = nullptr;
      this->f_packet.size = 0;

      int err = avcodec_decode_video2(this->f_video_encoding,
        this->f_frame, &got_picture,
        &this->f_packet);
      if (err >= 0)
      {
        this->f_pts += static_cast<int64_t>(this->stream_time_base_to_frame());
      }
    }

    // The cached frame is out of date, whether we managed to get a new
    // frame or not.
    this->current_image_memory = nullptr;

    if (!got_picture)
    {
      this->f_frame->data[0] = NULL;
    }

    return got_picture != 0;
  }

  // ==================================================================
  /*
  * @brief Get the current timestamp
  *
  * @return \b Current timestamp.
  */
  double current_pts() const
  {
    return this->f_pts * av_q2d(this->f_video_stream->time_base);
  }

  // ==================================================================
  /*
  * @brief Returns the double value to convert from a stream time base to
  *  a frame number
  */
  double stream_time_base_to_frame() const
  {
    if (this->f_video_stream->avg_frame_rate.num == 0.0)
    {
      return av_q2d(av_inv_q(av_mul_q(this->f_video_stream->time_base,
        this->f_video_stream->r_frame_rate)));
    }
    return av_q2d(
      av_inv_q(
        av_mul_q(this->f_video_stream->time_base, this->f_video_stream->avg_frame_rate)));
  }

  bool is_valid() const
  {
    return this->f_frame && this->f_frame->data[0];
  }

  // ==================================================================
  /*
  * @brief Return the current frame number
  *
  * @return \b Current frame number.
  */
  unsigned int frame_number() const
  {
    // Quick return if the stream isn't open.
    if (!this->is_valid())
    {
      return static_cast<unsigned int>(-1);
    }

    return static_cast<unsigned int>(
      (this->f_pts - this->f_start_time) / this->stream_time_base_to_frame()
      - static_cast<int>(this->f_frame_number_offset));
  }

  // ==================================================================
  /*
  * @brief Convert a ffmpeg pixel type to kwiver::vital::image_pixel_traits
  *
  * @return \b A kwiver::vital::image_pixel_traits
  */
  pixel_format pixel_format_from_ffmpeg(AVPixelFormat ffmpeg_format) const
  {
    switch (ffmpeg_format)
    {
    case AV_PIX_FMT_YUV420P:   return PIXEL_FORMAT_YUV_420P;
    case AV_PIX_FMT_YUYV422:   return PIXEL_FORMAT_YUYV_422;
    case AV_PIX_FMT_RGB24:     return PIXEL_FORMAT_RGB_24;
    case AV_PIX_FMT_BGR24:     return PIXEL_FORMAT_BGR_24;
    case AV_PIX_FMT_YUV422P:   return PIXEL_FORMAT_YUV_422P;
    case AV_PIX_FMT_YUV444P:   return PIXEL_FORMAT_YUV_444P;
#ifdef AV_PIX_FMT_RGBA
    case AV_PIX_FMT_RGBA:      return PIXEL_FORMAT_RGBA_32;
#endif
    case AV_PIX_FMT_YUV410P:   return PIXEL_FORMAT_YUV_410P;
    case AV_PIX_FMT_YUV411P:   return PIXEL_FORMAT_YUV_411P;
    case AV_PIX_FMT_RGB565:    return PIXEL_FORMAT_RGB_565;
    case AV_PIX_FMT_RGB555:    return PIXEL_FORMAT_RGB_555;
    case AV_PIX_FMT_GRAY8:     return PIXEL_FORMAT_MONO_8;
    case AV_PIX_FMT_PAL8:      return PIXEL_FORMAT_MONO_8;   //HACK: Treating 8-bit palette as greyscale image
    case AV_PIX_FMT_MONOWHITE: return PIXEL_FORMAT_MONO_1;
    case AV_PIX_FMT_MONOBLACK: return PIXEL_FORMAT_MONO_1;
    case AV_PIX_FMT_UYVY422:   return PIXEL_FORMAT_UYVY_422;
    case AV_PIX_FMT_UYYVYY411: return PIXEL_FORMAT_UYVY_411;
    default: break;
    }
    return PIXEL_FORMAT_UNKNOWN;
  }

}; // end of internal class.

// static open interlocking mutex
std::mutex ffmpeg_video_input::priv::open_mutex;


// ==================================================================
ffmpeg_video_input
::ffmpeg_video_input()
  : d( new priv() )
{
  attach_logger( "ffmpeg_video_input" ); // get appropriate logger
  d->logger = this->logger();
  ffmpeg_init();
}


ffmpeg_video_input
::~ffmpeg_video_input()
{
  this->close();
}


// ------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
ffmpeg_video_input
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  return config;
}


// ------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
ffmpeg_video_input
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.

  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);
}


// ------------------------------------------------------------------
bool
ffmpeg_video_input
::check_configuration(vital::config_block_sptr config) const
{
  bool retcode(true); // assume success

  return retcode;
}


// ------------------------------------------------------------------
void
ffmpeg_video_input
::open( std::string video_name )
{
  // Close any currently opened file
  this->close();

  d->video_path = video_name;

  {
    std::lock_guard< std::mutex > lock(d->open_mutex);

    if (!kwiversys::SystemTools::FileExists(d->video_path))
    {
      // Throw exception
      throw kwiver::vital::file_not_found_exception(video_name, "File not found");
    }

    if (!d->open(video_name))
    {
      throw kwiver::vital::video_runtime_exception("Video stream open failed for unknown reasons");
    }
  }
}


// ------------------------------------------------------------------
void
ffmpeg_video_input
::close()
{
  if (d->f_packet.data) {
    av_free_packet(&d->f_packet);  // free last packet
  }

  if (d->f_frame)
  {
    av_freep(&d->f_frame);
  }
  d->f_frame = nullptr;

  if (d->f_video_encoding && d->f_video_encoding->opaque)
  {
    av_freep(&d->f_video_encoding->opaque);
  }

  //d->num_frames_ = -2;
  //is_->contig_memory_ = 0;
  d->f_video_index = -1;
  d->f_data_index = -1;
  d->f_start_time = -1;
  d->video_path = "";
  //is_->metadata_.clear();
  if (d->f_video_stream)
  {
    avcodec_close(d->f_video_stream ->codec);
    d->f_video_stream = nullptr;
  }
  if (d->f_format_context)
  {
    avformat_close_input(&d->f_format_context);
    d->f_format_context = nullptr;
  }

  d->f_video_encoding = nullptr;
}


// ------------------------------------------------------------------
bool
ffmpeg_video_input
::next_frame( kwiver::vital::timestamp& ts,
              uint32_t timeout )
{
  if (!d->is_opened())
  {
    throw vital::file_not_read_exception(d->video_path, "Video not open");
  }

  bool ret = d->advance();
  if (ret)
  {
    ts.set_frame(d->frame_number());
  }
  return ret;
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
ffmpeg_video_input
::frame_image( )
{
  // Quick return if the stream isn't valid
  if (!d->is_valid())
  {
    return nullptr;
  }

  AVCodecContext* enc = d->f_format_context->streams[d->f_video_index]->codec;

  // If we have not already converted this frame, try to convert it

  if (!d->current_image_memory && d->f_frame->data[0] != 0)
  {
    int width = enc->width;
    int height = enc->height;

    // If the pixel format is not recognized by then convert the data into RGB_24
    ffmpeg_video_input::priv::pixel_format format = d->pixel_format_from_ffmpeg(enc->pix_fmt);
    if (format == ffmpeg_video_input::priv::PIXEL_FORMAT_UNKNOWN)
    {
      int size = width * height * 3;
      if (!d->current_image_memory || size != d->current_image_memory->size())
      {
        d->current_image_memory = vital::image_memory_sptr(new vital::image_memory(size));
      }

      d->f_software_context = sws_getCachedContext(
        d->f_software_context,
        width, height, enc->pix_fmt,
        width, height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        NULL, NULL, NULL);

      if (!d->f_software_context)
      {
        LOG_ERROR(this->logger(), "Couldn't create conversion context");
        return nullptr;
      }

      AVPicture rgb_frame;
      avpicture_fill(&rgb_frame, (uint8_t*)d->current_image_memory->data(), AV_PIX_FMT_RGB24, width, height);

      int w_step = 1;
      int h_step = width;
      int d_step = width*height;

      sws_scale(d->f_software_context,
        d->f_frame->data, d->f_frame->linesize,
        0, height,
        rgb_frame.data, rgb_frame.linesize);

      vital::image image(
        d->current_image_memory,
        d->current_image_memory->data(),
        width, height, 3,
        3, 3*width, 3*width*height,
        vital::image_pixel_traits_of<unsigned char>());

      d->current_image = std::make_shared<vital::simple_image_container>(vital::simple_image_container(image));
    }
    else
    {
      // Test for contiguous memory.  Sometimes FFMPEG uses scanline buffers larger
      // than the image width.  The extra memory is used in optimized decoding routines.
      // This leads to a segmented image buffer, not supported by vidl.
      AVPicture test_frame;
      avpicture_fill(&test_frame, d->f_frame->data[0], enc->pix_fmt, width, height);
      if (test_frame.data[1] == d->f_frame->data[1] &&
        test_frame.data[2] == d->f_frame->data[2] &&
        test_frame.linesize[0] == d->f_frame->linesize[0] &&
        test_frame.linesize[1] == d->f_frame->linesize[1] &&
        test_frame.linesize[2] == d->f_frame->linesize[2])
      {
      vital::image image(
          d->f_frame->data[0],
          width, height, 3,
          3, 3 * width, 3 * width*height//,
          //image_pixel_from_traits_macro from fmt ?
        );
      d->current_image = std::make_shared<vital::simple_image_container>(vital::simple_image_container(image));
      }
      // Copy the image into contiguous memory.
      else
      {
        if (!d->current_image_memory)
        {
          int size = avpicture_get_size(enc->pix_fmt, width, height);
          d->current_image_memory = vital::image_memory_sptr(new vital::image_memory(size));
        }
        avpicture_fill(&test_frame, (uint8_t*)d->current_image_memory->data(), enc->pix_fmt, width, height);
        av_picture_copy(&test_frame, (AVPicture*)d->f_frame, enc->pix_fmt, width, height);

        vital::image image(
          d->current_image_memory,
          nullptr,
          width, height, 3,
          3, 3 * width, 3 * width*height //,
          //image_pixel_from_traits_macro from fmt ?
          );
        d->current_image = std::make_shared<vital::simple_image_container>(vital::simple_image_container(image));

      }
    }
  }

  return d->current_image;
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
ffmpeg_video_input
::frame_metadata()
{
  return kwiver::vital::metadata_vector();
}


// ------------------------------------------------------------------
bool
ffmpeg_video_input
::end_of_video() const
{
  return false;
}


// ------------------------------------------------------------------
bool
ffmpeg_video_input
::good() const
{
  return d->is_valid();
}

} } } // end namespaces
