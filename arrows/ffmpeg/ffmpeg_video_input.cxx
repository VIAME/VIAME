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
#include <vital/exceptions/metadata.h>
#include <vital/exceptions/video.h>
#include <vital/util/tokenize.h>
#include <vital/klv/convert_metadata.h>
#include <vital/klv/misp_time.h>
#include <vital/klv/klv_data.h>

#include <arrows/vxl/image_container.h>

#include <kwiversys/SystemTools.hxx>

#include <mutex>
#include <memory>
#include <vector>
#include <sstream>

extern "C" {
#if FFMPEG_IN_SEVERAL_DIRECTORIES
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#else
#include <ffmpeg/avcodec.h>
#include <ffmpeg/avformat.h>
#include <ffmpeg/swscale.h>
#endif
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
    format_context(nullptr),
    video_index(-1),
    data_index(-1),
    video_encoding(nullptr),
    video_stream(nullptr),
    frame(nullptr),
    start_time(-1),
    frame_number_offset(0)
  {}


  AVFormatContext* format_context;
  int video_index;
  int data_index;
  AVCodecContext* video_encoding;
  AVStream* video_stream;
  AVFrame* frame;

  //: Start time of the stream, to offset the pts when computing the frame number.
  //  (in stream time base)
  int64_t start_time;

  //: Some codec/file format combinations need a frame number offset.
  // These codecs have a delay between reading packets and generating frames.
  unsigned frame_number_offset;

}; // end of internal class.

// static open interlocking mutex
//std::mutex ffmpeg_video_input::priv::s_open_mutex;


// ==================================================================
ffmpeg_video_input
::ffmpeg_video_input()
  : d( new priv() )
{
  attach_logger( "ffmpeg_video_input" ); // get appropriate logger
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

  // Open the file
  int err = avformat_open_input(&d->format_context, video_name.c_str(), NULL, NULL);
  if (err != 0)
  {
    LOG_ERROR(this->logger(), "Error " << err << " trying to open " << video_name );
    return;
  }

  // Get the stream information by reading a bit of the file
  if (avformat_find_stream_info(d->format_context, NULL) < 0)
  {
    return;
  }

  // Find a video stream, and optionally a data stream.
  // Use the first ones we find.
  d->video_index = -1;
  d->data_index = -1;
  AVCodecContext* codec_context_origin = NULL;
  for (unsigned i = 0; i < d->format_context->nb_streams; ++i)
  {
    AVCodecContext *const enc = d->format_context->streams[i]->codec;
    if (enc->codec_type == AVMEDIA_TYPE_VIDEO && d->video_index < 0)
    {
      d->video_index = i;
      codec_context_origin = enc;
    }
    else if (enc->codec_type == AVMEDIA_TYPE_DATA && d->data_index < 0)
    {
      d->data_index = i;
    }
  }

  if (d->video_index < 0)
  {
    LOG_ERROR(this->logger(), "Error: could not find a video stream in " << video_name);
    return;
  }

  if (d->data_index < 0)
  {
    LOG_INFO(this->logger(), "No data stream available, using AVMEDIA_TYPE_UNKNOWN stream instead");
    // Fallback for the DATA stream if incorrectly coded as UNKNOWN.
    for (unsigned i = 0; i < d->format_context->nb_streams; ++i)
    {
      AVCodecContext *enc = d->format_context->streams[i]->codec;
      if (enc->codec_type == AVMEDIA_TYPE_UNKNOWN)
      {
        d->data_index = i;
      }
    }
  }

  assert(codec_context_origin);
  av_dump_format(d->format_context, 0, video_name.c_str(), 0);

  // Open the stream
  AVCodec* codec = avcodec_find_decoder(codec_context_origin->codec_id);
  if (!codec)
  {
    LOG_ERROR(this->logger(),
      "Error: Codec " << codec_context_origin->codec_descriptor
      << " (" << codec_context_origin->codec_id << ") not found");
    return;
  }

  // Copy context
  d->video_encoding = avcodec_alloc_context3(codec);
  if (avcodec_copy_context(d->video_encoding, codec_context_origin) != 0)
  {
    LOG_ERROR(this->logger(), "Error: Could not copy codec " << d->video_encoding->codec_name);
    return;
  }

  // Open codec
  if (avcodec_open2(d->video_encoding, codec, NULL) < 0)
  {
    LOG_ERROR(this->logger(), "Error: Could not open codec " << d->video_encoding->codec_name);
    return;
  }

  d->video_stream = d->format_context->streams[d->video_index];
  d->frame = av_frame_alloc();

  if (d->video_stream->start_time == int64_t(1) << 63)
  {
    d->start_time = 0;
  }
  else
  {
    d->start_time = d->video_stream->start_time;
  }

  // The MPEG 2 codec has a latency of 1 frame when encoded in an AVI
  // stream, so the pts of the last packet (stored in pts) is
  // actually the next frame's pts.
  if (d->video_stream->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO &&
    std::string("avi") == d->format_context->iformat->name)
  {
    d->frame_number_offset = 1;
  }

  /*// Not sure if this does anything, but no harm either
  av_init_packet(&is_->packet_);
  is_->packet_.data = 0;
  is_->packet_.size = 0;*/
}


// ------------------------------------------------------------------
void
ffmpeg_video_input
::close()
{
  /*if (is_->packet_.data) {
    av_free_packet(&is_->packet_);  // free last packet
  }*/

  if (d->frame)
  {
    av_freep(&d->frame);
  }

  if (d->video_encoding && d->video_encoding->opaque)
  {
    av_freep(&d->video_encoding->opaque);
  }

  //d->num_frames_ = -2;
  //is_->contig_memory_ = 0;
  d->video_index = -1;
  d->data_index = -1;
  //is_->metadata_.clear();
  if (d->video_stream)
  {
    avcodec_close(d->video_stream ->codec);
    d->video_stream = nullptr;
  }
  if (d->format_context)
  {
    avformat_close_input(&d->format_context);
    d->format_context = nullptr;
  }

  d->video_encoding = nullptr;
}


// ------------------------------------------------------------------
bool
ffmpeg_video_input
::next_frame( kwiver::vital::timestamp& ts,
              uint32_t timeout )
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
ffmpeg_video_input
::frame_image( )
{
  return nullptr;
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
  return d->frame && d->frame->data[0];
}

} } } // end namespaces
