// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation file for video input using FFmpeg.
 */

#include "ffmpeg_init.h"
#include "ffmpeg_video_input.h"

#include <vital/types/timestamp.h>
#include <vital/exceptions/io.h>
#include <vital/exceptions/video.h>
#include <vital/klv/convert_metadata.h>
#include <vital/klv/misp_time.h>
#include <vital/klv/klv_data.h>
#include <vital/util/tokenize.h>
#include <vital/types/image_container.h>
#include <vital/vital_config.h>

#include <kwiversys/SystemTools.hxx>

#include <deque>
#include <mutex>
#include <memory>
#include <vector>
#include <sstream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavfilter/avfiltergraph.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
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
    f_format_context(avformat_alloc_context()),
    f_video_index(-1),
    f_data_index(-1),
    f_video_encoding(nullptr),
    f_video_stream(nullptr),
    f_frame(nullptr),
    f_filtered_frame(nullptr),
    f_packet(nullptr),
    f_software_context(nullptr),
    f_filter_graph(nullptr),
    f_filter_sink_context(nullptr),
    f_filter_src_context(nullptr),
    f_start_time(-1),
    f_backstep_size(-1),
    f_frame_number_offset(0),
    video_path(""),
    filter_desc("yadif=deint=1"),
    metadata(0),
    frame_advanced(false),
    end_of_video(true),
    number_of_frames(0),
    collected_all_metadata(false),
    estimated_num_frames(false)
  { }

  // f_* variables are FFmpeg specific

  AVFormatContext* f_format_context;
  int f_video_index;
  int f_data_index;
  AVCodecContext* f_video_encoding;
  AVStream* f_video_stream;
  AVFrame* f_frame;
  AVFrame* f_filtered_frame;
  AVPacket* f_packet;
  SwsContext* f_software_context;
  AVFilterGraph* f_filter_graph;
  AVFilterContext *f_filter_sink_context;
  AVFilterContext *f_filter_src_context;

  // Start time of the stream, to offset the pts when computing the frame number.
  // (in stream time base)
  int64_t f_start_time;

  // Presentation timestamp (in stream time base)
  int64_t f_pts;

  // Number of frames to back step when seek fails to land on frame before request
  int64_t f_backstep_size;

  // Some codec/file format combinations need a frame number offset.
  // These codecs have a delay between reading packets and generating frames.
  unsigned f_frame_number_offset;

  // Name of video we opened
  std::string video_path;

  // FFMPEG filter description string
  // What you put after -vf in the ffmpeg command line tool
  std::string filter_desc;

  // the buffer of metadata from the data stream
  std::deque<uint8_t> metadata;

  // metadata converter object
  kwiver::vital::convert_metadata converter;

  /**
   * Storage for the metadata map.
   */
  vital::metadata_map::map_metadata_t metadata_map;

  static std::mutex open_mutex;

  // For logging in priv methods
  vital::logger_handle_t logger;

  // Current image frame.
  vital::image_memory_sptr current_image_memory;
  kwiver::vital::image_container_sptr current_image;

  // local state
  bool frame_advanced;
  bool end_of_video;
  size_t number_of_frames;
  bool collected_all_metadata;
  bool estimated_num_frames;

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
    AVCodecParameters* codec_param_origin = NULL;
    for (unsigned i = 0; i < this->f_format_context->nb_streams; ++i)
    {
      AVCodecParameters* params = this->f_format_context->streams[i]->codecpar;
      if (params->codec_type == AVMEDIA_TYPE_VIDEO && this->f_video_index < 0)
      {
        this->f_video_index = i;
        codec_param_origin = params;
      }
      else if (params->codec_type == AVMEDIA_TYPE_DATA && this->f_data_index < 0)
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
      LOG_INFO(this->logger, "No data stream available");
      // Fallback for the DATA stream if incorrectly coded as UNKNOWN.
      for (unsigned i = 0; i < this->f_format_context->nb_streams; ++i)
      {
        AVCodecParameters* params = this->f_format_context->streams[i]->codecpar;
        if (params->codec_type == AVMEDIA_TYPE_UNKNOWN)
        {
          this->f_data_index = i;
          LOG_INFO(this->logger, "Using AVMEDIA_TYPE_UNKNOWN stream as a data stream");
        }
      }
    }

    av_dump_format(this->f_format_context, 0, this->video_path.c_str(), 0);

    // Open the stream
    AVCodec* codec = avcodec_find_decoder(codec_param_origin->codec_id);
    if (!codec)
    {
      LOG_ERROR(this->logger,
        "Error: Codec " << avcodec_descriptor_get(codec_param_origin->codec_id)
        << " (" << codec_param_origin->codec_id << ") not found");
      return false;
    }

    // Copy context
    this->f_video_encoding = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(this->f_video_encoding, codec_param_origin) > 0)
    {
      LOG_ERROR(this->logger,
        "Error: Could not fill codec context " << this->f_video_encoding->codec_id);
      return false;
    }

    // Open codec
    if (avcodec_open2(this->f_video_encoding, codec, NULL) < 0)
    {
      LOG_ERROR(this->logger, "Error: Could not open codec " << this->f_video_encoding->codec_id);
      return false;
    }

    bool empty_filter = filter_desc.empty() ||
                        std::all_of(filter_desc.begin(),
                                    filter_desc.end(), isspace);
    if (!empty_filter && !this->init_filters(filter_desc))
    {
      return false;
    }

    // Use group of picture (GOP) size for seek back step if avaiable
    if ( this->f_video_encoding->gop_size > 0 )
    {
      this->f_backstep_size = this->f_video_encoding->gop_size;
    }
    else
    {
      // If GOP size not available use 12 which is a common GOP size.
      this->f_backstep_size = 12;
    }

    this->f_video_stream = this->f_format_context->streams[this->f_video_index];
    this->f_frame = av_frame_alloc();
    this->f_filtered_frame = av_frame_alloc();
    this->f_packet = av_packet_alloc();

    // The MPEG 2 codec has a latency of 1 frame when encoded in an AVI
    // stream, so the pts of the last packet (stored in pts) is
    // actually the next frame's pts.
    if (this->f_video_stream->codecpar->codec_id == AV_CODEC_ID_MPEG2VIDEO &&
      std::string("avi") == this->f_format_context->iformat->name)
    {
      this->f_frame_number_offset = 1;
    }

    // Advance to first valid frame to get start time
    this->f_start_time = 0;
    if ( this->advance() )
    {
        this->f_start_time = this->f_pts;
    }
    else
    {
        LOG_ERROR(this->logger, "Error: failed to find valid frame to set start time");
        this->f_start_time = -1;
        return false;
    }

    // Now seek back to the start of the video
    auto seek_rslt = av_seek_frame( this->f_format_context,
                                    this->f_video_index,
                                    0,
                                    AVSEEK_FLAG_BACKWARD );
    avcodec_flush_buffers( this->f_video_encoding );
    if (seek_rslt < 0 )
    {
        LOG_ERROR(this->logger,
                  "Error: failed to return to start after setting start time");
        return false;
    }
    this->frame_advanced = false;
    this->f_frame->data[0] = NULL;
    return true;
  }

  // ==================================================================
  /*
  * @brief Close the current video.
  */
  void close()
  {
    this->f_video_index = -1;
    this->f_data_index = -1;
    this->f_start_time = -1;

    if (this->f_video_stream)
    {
      this->f_video_stream = nullptr;
    }

    av_frame_free(&this->f_frame);
    av_frame_free(&this->f_filtered_frame);
    av_packet_free(&this->f_packet);
    avformat_close_input(&this->f_format_context);
    avformat_free_context(this->f_format_context);
    avcodec_free_context(&this->f_video_encoding);
    avfilter_graph_free(&this->f_filter_graph);
  }

  // ==================================================================
  /*
  * @brief Initialize the filter graph
  */
  bool init_filters(std::string const& filters_desc)
  {
    auto deleter = [](AVFilterInOut** ptr)
    {
      avfilter_inout_free(ptr);
      delete[] ptr;
    };
    using AVFilterInOut_ptr = std::unique_ptr<AVFilterInOut*, decltype(deleter)>;
    char args[512];
    int ret = 0;
    AVFilter *buffersrc = avfilter_get_by_name("buffer");
    AVFilter *buffersink = avfilter_get_by_name("buffersink");
    AVFilterInOut_ptr outputs(new AVFilterInOut*[1], deleter);
    *outputs.get() = avfilter_inout_alloc();
    AVFilterInOut_ptr inputs(new AVFilterInOut*[1], deleter);
    *inputs.get() = avfilter_inout_alloc();

    AVRational time_base = f_format_context->streams[f_video_index]->time_base;
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGB24, AV_PIX_FMT_GRAY8,
                                      AV_PIX_FMT_NONE };
    this->f_filter_graph = avfilter_graph_alloc();
    if (!outputs || !inputs || !f_filter_graph)
    {
      LOG_ERROR(this->logger, "Failed to alloation filter graph");
      return false;
    }
    // Buffer video source
    // The decoded frames from the decoder will be inserted here.
    snprintf(args, sizeof(args),
      "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
      f_video_encoding->width, f_video_encoding->height,
      f_video_encoding->pix_fmt,
      time_base.num, time_base.den,
      f_video_encoding->sample_aspect_ratio.num,
      f_video_encoding->sample_aspect_ratio.den);
    ret = avfilter_graph_create_filter(&f_filter_src_context, buffersrc, "in",
                                       args, NULL, f_filter_graph);
    if (ret < 0)
    {
      LOG_ERROR(this->logger, "Cannot create buffer source");
      return false;
    }
    // Buffer video sink
    // To terminate the filter chain.
    ret = avfilter_graph_create_filter(&f_filter_sink_context,
                                       buffersink, "out",
                                       NULL, NULL, f_filter_graph);
    if (ret < 0)
    {
      LOG_ERROR(this->logger, "Cannot create buffer sink");
      return false;
    }
    ret = av_opt_set_int_list(f_filter_sink_context, "pix_fmts", pix_fmts,
                              AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0)
    {
      LOG_ERROR(this->logger, "Cannot set output pixel format");
      return false;
    }

    // Set the endpoints for the filter graph. The filter_graph will
    // be linked to the graph described by filters_desc.

    // The buffer source output must be connected to the input pad of
    // the first filter described by filters_desc; since the first
    // filter input label is not specified, it is set to "in" by
    // default.
    (*outputs)->name = av_strdup("in");
    (*outputs)->filter_ctx = f_filter_src_context;
    (*outputs)->pad_idx = 0;
    (*outputs)->next = NULL;

    // The buffer sink input must be connected to the output pad of
    // the last filter described by filters_desc; since the last
    // filter output label is not specified, it is set to "out" by
    // default.
    (*inputs)->name = av_strdup("out");
    (*inputs)->filter_ctx = f_filter_sink_context;
    (*inputs)->pad_idx = 0;
    (*inputs)->next = NULL;

    if (avfilter_graph_parse_ptr(f_filter_graph, filters_desc.c_str(),
                                 inputs.get(), outputs.get(), NULL) < 0)
    {
      LOG_ERROR(this->logger, "Failed to parse AV filter graph");
      return false;
    }

    if (avfilter_graph_config(f_filter_graph, NULL) < 0)
    {
      LOG_ERROR(this->logger, "Failed to configure AV filter graph");
      return false;
    }

    return true;
  }

  // ==================================================================
  /*
  * @brief Advance to the next frame (but don't acquire an image).
  *
  * @return \b true if video was valid and we found a frame.
  */
  bool advance()
  {
    // Quick return if the file isn't open.
    if (!this->is_opened())
    {
      this->frame_advanced = false;
      return false;
    }

    this->frame_advanced = false;

    // clear the metadata from the previous frame
    this->metadata.clear();

    while (!this->frame_advanced &&
           av_read_frame(this->f_format_context, this->f_packet) >= 0)
    {
      // Make sure that the packet is from the actual video stream.
      if (this->f_packet->stream_index == this->f_video_index)
      {
        int err = avcodec_send_packet(this->f_video_encoding, this->f_packet);
        if (err < 0)
        {
          LOG_ERROR(this->logger, "Error sending packet to decoder");
          return false;
        }

        err = avcodec_receive_frame(this->f_video_encoding, this->f_frame);

        // Ignore the frame and move to the next
        if (err == AVERROR_INVALIDDATA)
        {
          av_packet_unref(this->f_packet);
          continue;
        }
        if (err < 0)
        {
          LOG_ERROR(this->logger, "Error decoding packet");
          av_packet_unref(this->f_packet);
          return false;
        }

        this->f_pts = av_frame_get_best_effort_timestamp(this->f_frame);
        if (this->f_pts == AV_NOPTS_VALUE)
        {
          this->f_pts = 0;
        }

        this->frame_advanced = true;
      }

      // grab the metadata from this packet if from the metadata stream
      else if (this->f_packet->stream_index == this->f_data_index)
      {
        this->metadata.insert(this->metadata.end(), this->f_packet->data,
          this->f_packet->data + this->f_packet->size);
      }

      // De-reference previous packet
      av_packet_unref(this->f_packet);
    }

    // The cached frame is out of date, whether we managed to get a new
    // frame or not.
    this->current_image_memory = nullptr;

    if (!this->frame_advanced)
    {
      this->f_frame->data[0] = NULL;
    }

    return this->frame_advanced;
  }

  // ==================================================================
  /*
  * @brief Seek to a specific frame
  *
  * @return \b true if video was valid and we found a frame.
  */
  bool seek( uint64_t frame )
  {
    // Time for frame before requested frame. The frame before is requested so
    // advance will called at least once in case the request lands on a keyframe.
    int64_t frame_ts = (static_cast<int>(f_frame_number_offset) + frame - 1) *
      this->stream_time_base_to_frame() + this->f_start_time;

    bool advance_successful = false;
    do
    {
      auto seek_rslt = av_seek_frame( this->f_format_context,
                                      this->f_video_index, frame_ts,
                                      AVSEEK_FLAG_BACKWARD );
      avcodec_flush_buffers( this->f_video_encoding );

      if ( seek_rslt < 0 )
      {
        return false;
      }

      advance_successful = this->advance();

      // Continue to make seek request further back until we land at a frame
      // that is before the requested frame.
      frame_ts -= this->f_backstep_size * this->stream_time_base_to_frame();
    }
    while( this->frame_number() > frame - 1 || !advance_successful );

    // Now advance forward until we reach the requested frame.
    while( this->frame_number() < frame - 1 )
    {
      if ( !this->advance() )
      {
        return false;
      }

      if ( this->frame_number() > frame -1 )
      {
        LOG_ERROR( this->logger, "seek went past requested frame." );
        return false;
      }
    }

    return true;
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

  void set_default_metadata(kwiver::vital::metadata_sptr md)
  {
    // Add frame number to timestamp
    kwiver::vital::timestamp ts;
    ts.set_frame( this->frame_number() );
    md->set_timestamp( ts );

    // Add file name/uri
    md->add< vital::VITAL_META_VIDEO_URI >( video_path );

    // Mark whether the frame is a key frame
    if ( this->f_frame->key_frame > 0 )
    {
      md->add< vital::VITAL_META_VIDEO_KEY_FRAME >( true );
    }
    else
    {
      md->add< vital::VITAL_META_VIDEO_KEY_FRAME >( false );
    }
  }

  kwiver::vital::metadata_vector current_metadata()
  {
    kwiver::vital::metadata_vector retval;

    // Copy the current raw metadata
    std::deque<uint8_t> md_buffer = this->metadata;

    kwiver::vital::klv_data klv_packet;

    // If we have collected enough of the stream to make a KLV packet
    while ( klv_pop_next_packet( md_buffer, klv_packet ) )
    {
      auto meta = std::make_shared<kwiver::vital::metadata>();

      try
      {
        converter.convert( klv_packet, *(meta) );
      }
      catch ( kwiver::vital::metadata_exception const& e )
      {
        LOG_WARN( this->logger, "Metadata exception: " << e.what() );
        continue;
      }

      // If the metadata was even partially decided, then add to the list.
      if ( ! meta->empty() )
      {
        set_default_metadata( meta );

        retval.push_back( meta );
      } // end valid metadata packet.
    } // end while

    // if no metadata from the stream, add a basic metadata item
    if ( retval.empty() )
    {
      auto meta = std::make_shared<kwiver::vital::metadata>();
      set_default_metadata( meta );

      retval.push_back(meta);
    }

    return retval;
  }

  // ==================================================================
  /*
  * @brief Loop over all frames to collect metadata
  */
  void collect_all_metadata()
  {
    // is stream open?
    if ( ! this->is_opened() )
    {
      VITAL_THROW( vital::file_not_read_exception, video_path, "Video not open" );
    }

    if ( !collected_all_metadata )
    {
      std::lock_guard< std::mutex > lock( open_mutex );

      auto initial_frame_number = this->frame_number();

      if ( !frame_advanced && !end_of_video )
      {
        initial_frame_number = 0;
      }

      // Add metadata for current frame
      if ( frame_advanced )
      {
        this->metadata_map.insert(
          std::make_pair( this->frame_number(), this->current_metadata() ) );
      }

      // Advance video stream to end
      while ( this->advance() )
      {
        this->metadata_map.insert(
          std::make_pair( this->frame_number(), this->current_metadata() ) );
      }

      // Close and reopen to reset
      this->close();
      this->open( video_path );

      // Advance back to original frame number
      unsigned int frame_num = 0;
      while ( frame_num < initial_frame_number && this->advance() )
      {
        ++frame_num;
        this->metadata_map.insert(
          std::make_pair( this->frame_number(), this->current_metadata() ) );
      }

      collected_all_metadata = true;
    }
  }

  // ==================================================================
  /*
  * @brief Seek to the end of the video to estimate number of frames
  */
  void estimate_num_frames()
  {
    // is stream open?
    if (!this->is_opened())
    {
      VITAL_THROW(vital::file_not_read_exception, video_path, "Video not open");
    }

    if ( !estimated_num_frames )
    {
      std::lock_guard< std::mutex > lock(open_mutex);

      auto initial_frame_number = this->frame_number();

      if (!frame_advanced && !end_of_video)
      {
        initial_frame_number = 0;
      }

      bool advance_successful = false;
      // Seek to as close as possibly to the end of the video
      int64_t frame_ts = this->f_video_stream->duration + this->f_start_time;
      do
      {
        auto seek_rslt = av_seek_frame(this->f_format_context,
                                       this->f_video_index,
                                       frame_ts,
                                       AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(this->f_video_encoding);

        if (seek_rslt < 0)
        {
          break;
        }

        advance_successful = this->advance();

        // Continue to make seek request further back until we land at a valid frame
        frame_ts -= this->f_backstep_size * this->stream_time_base_to_frame();
      } while (!advance_successful);

      LOG_DEBUG(this->logger, "Seeked to near end frame: " << this->frame_number());

      // Step through the end of the video to find the last valid frame number
      do
      {
        number_of_frames = this->frame_number();
      }
      while (this->advance());
      // The number of frames is one greater than the last valid frame number
      ++number_of_frames;

      LOG_DEBUG(this->logger, "Found " << number_of_frames << " video frames");

      // Set this flag so we do not have a count frames next time
      estimated_num_frames = true;

      // Return the video to its state before seeking to the end
      if (initial_frame_number == 0)
      {
        // Close and reopen to reset
        this->close();
        this->open(video_path);
      }
      else
      {
        this->seek(initial_frame_number);
      }
    }
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

  this->set_capability(vital::algo::video_input::HAS_EOV, true);
  this->set_capability(vital::algo::video_input::HAS_FRAME_NUMBERS, true);
  this->set_capability(vital::algo::video_input::HAS_FRAME_DATA, true);
  this->set_capability(vital::algo::video_input::HAS_METADATA, false);

  this->set_capability(vital::algo::video_input::HAS_FRAME_TIME, false);
  this->set_capability(vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME, false);
  this->set_capability(vital::algo::video_input::HAS_TIMEOUT, false);
  this->set_capability(vital::algo::video_input::IS_SEEKABLE, true);

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

  config->set_value("filter_desc", d->filter_desc,
    "A string describing the libavfilter pipeline to apply when reading "
    "the video.  Only filters that operate on each frame independently "
    "will currently work.  The default \"yadif=deint=1\" filter applies "
    "deinterlacing only to frames which are interlaced.  "
    "See details at https://ffmpeg.org/ffmpeg-filters.html");

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

  d->filter_desc = config->get_value<std::string>("filter_desc", d->filter_desc);
}

// ------------------------------------------------------------------
bool
ffmpeg_video_input
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
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
      VITAL_THROW( kwiver::vital::file_not_found_exception, video_name, "File not found");
    }

    if (!d->open(video_name))
    {
      VITAL_THROW( kwiver::vital::video_runtime_exception, "Video stream open failed for unknown reasons");
    }
    this->set_capability(vital::algo::video_input::HAS_METADATA,
                         d->f_data_index >= 0);
    d->end_of_video = false;
  }
}

// ------------------------------------------------------------------
void
ffmpeg_video_input
::close()
{
  d->close();

  d->video_path = "";
  d->frame_advanced = false;
  d->end_of_video = true;
  d->number_of_frames = 0;
  d->collected_all_metadata = false;
  d->estimated_num_frames = false;
  d->metadata.clear();
}

// ------------------------------------------------------------------
bool
ffmpeg_video_input
::next_frame( kwiver::vital::timestamp& ts,
              VITAL_UNUSED uint32_t timeout )
{
  if (!d->is_opened())
  {
    VITAL_THROW( vital::file_not_read_exception, d->video_path, "Video not open");
  }

  bool ret = d->advance();

  d->end_of_video = !ret;
  if (ret)
  {
    ts = this->frame_timestamp();
  };
  return ret;
}

// ------------------------------------------------------------------
bool ffmpeg_video_input::seek_frame(kwiver::vital::timestamp& ts,
  kwiver::vital::timestamp::frame_t frame_number,
  uint32_t timeout)
{
  // Quick return if the stream isn't open.
  if (!d->is_opened())
  {
    VITAL_THROW( vital::file_not_read_exception, d->video_path, "Video not open");
    return false;
  }
  if (frame_number <= 0)
  {
    return false;
  }

  if (timeout != 0)
  {
    LOG_WARN(this->logger(), "Timeout argument is not supported.");
  }

  bool ret = d->seek( frame_number );
  d->end_of_video = !ret;
  if (ret)
  {
    ts = this->frame_timestamp();
  };
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

  AVCodecParameters* params = d->f_format_context->streams[d->f_video_index]->codecpar;

  // If we have not already converted this frame, try to convert it
  if (!d->current_image_memory && d->f_frame->data[0] != 0)
  {
    int width = params->width;
    int height = params->height;
    int depth = 3;
    vital::image_pixel_traits pixel_trait = vital::image_pixel_traits_of<unsigned char>();
    bool direct_copy;
    AVFrame* frame = d->f_frame;

    if (d->f_filter_src_context && d->f_filter_sink_context)
    {
      // Since we are only reading one frame at a time we need to push this
      // frame into the filter pipeline repeatedly until the same frame comes
      // out the other side.
      int ret = AVERROR(EAGAIN);
      while (ret == AVERROR(EAGAIN) ||
        d->f_frame->best_effort_timestamp != d->f_filtered_frame->best_effort_timestamp)
      {
        // Push the decoded frame into the filter graph
        if (av_buffersrc_add_frame_flags(d->f_filter_src_context, d->f_frame,
          AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
        {
          LOG_ERROR(this->logger(), "Error while feeding the filter graph");
          return nullptr;
        }
        // Pull a filtered frame from the filter graph
        ret = av_buffersink_get_frame(d->f_filter_sink_context, d->f_filtered_frame);
        if (ret == AVERROR_EOF)
        {
          return nullptr;
        }
      }
      frame = d->f_filtered_frame;
    }

    AVPixelFormat pix_fmt = static_cast<AVPixelFormat>(frame->format);
    switch (pix_fmt)
    {
      case AV_PIX_FMT_GRAY8:
      {
        depth = 1;
        direct_copy = true;
        break;
      }
      case AV_PIX_FMT_RGB24:
      {
        depth = 3;
        direct_copy = true;
        break;
      }
      case AV_PIX_FMT_RGBA:
      {
        depth = 4;
        direct_copy = true;
        break;
      }
      case AV_PIX_FMT_MONOWHITE:
      case AV_PIX_FMT_MONOBLACK:
      {
        depth = 1;
        pixel_trait = vital::image_pixel_traits_of<bool>();
        direct_copy = true;
        break;
      }
      default:
      {
        direct_copy = false;
      }
    }

    if (direct_copy)
    {
      int size = av_image_get_buffer_size(pix_fmt, width, height, 1);
      d->current_image_memory = vital::image_memory_sptr(new vital::image_memory(size));

      AVFrame picture;
      av_image_fill_arrays(picture.data, picture.linesize,
                           static_cast<uint8_t *>(d->current_image_memory->data()),
                           pix_fmt, width, height, 1);
      auto framedata = const_cast<const uint8_t **>(frame->data);
      av_image_copy(picture.data, picture.linesize,
                    framedata, frame->linesize,
                    pix_fmt, width, height);
    }
    else
    // If the pixel format is not recognized by then convert the data into RGB_24
    {
      int size = width * height * depth;
      d->current_image_memory = std::make_shared<vital::image_memory>(size);

      d->f_software_context = sws_getCachedContext(
        d->f_software_context,
        width, height, pix_fmt,
        width, height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        NULL, NULL, NULL);

      if (!d->f_software_context)
      {
        LOG_ERROR(this->logger(), "Couldn't create conversion context");
        return nullptr;
      }

      AVFrame rgb_frame;
      av_image_fill_arrays(rgb_frame.data, rgb_frame.linesize,
                           static_cast<uint8_t*>(d->current_image_memory->data()),
                           AV_PIX_FMT_RGB24, width, height, 1);

      sws_scale(d->f_software_context,
        frame->data, frame->linesize,
        0, height,
        rgb_frame.data, rgb_frame.linesize);
    }

    vital::image image(
      d->current_image_memory,
      d->current_image_memory->data(),
      width, height, depth,
      depth, depth * width, 1
    );
    d->current_image = std::make_shared<vital::simple_image_container>(vital::simple_image_container(image));
  }

  return d->current_image;
}

// ------------------------------------------------------------------
kwiver::vital::timestamp
ffmpeg_video_input
::frame_timestamp() const
{
  if (!this->good())
  {
    return {};
  }

  // We don't always have all components of a timestamp, so start with
  // an invalid TS and add the data we have.
  kwiver::vital::timestamp ts;
  ts.set_frame(d->frame_number() + d->f_frame_number_offset + 1);

  return ts;
}

// ------------------------------------------------------------------
kwiver::vital::metadata_vector
ffmpeg_video_input
::frame_metadata()
{
  return d->current_metadata();
}

// ------------------------------------------------------------------
kwiver::vital::metadata_map_sptr
ffmpeg_video_input
::metadata_map()
{
  d->collect_all_metadata();

  return std::make_shared<kwiver::vital::simple_metadata_map>(d->metadata_map);
}

// ------------------------------------------------------------------
bool
ffmpeg_video_input
::end_of_video() const
{
  return d->end_of_video;
}

// ------------------------------------------------------------------
bool
ffmpeg_video_input
::good() const
{
  return d->is_valid() && d->frame_advanced;
}

// ------------------------------------------------------------------
bool
ffmpeg_video_input
::seekable() const
{
  return true;
}

// ------------------------------------------------------------------
size_t
ffmpeg_video_input
::num_frames() const
{
  d->estimate_num_frames();

  return d->number_of_frames;
}

} } } // end namespaces
