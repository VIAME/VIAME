/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * \brief Implementation file for video input using VXL methods.
 */

#include "vidl_ffmpeg_video_input.h"

#include <vital/types/timestamp.h>
#include <vital/exceptions/io.h>
#include <vital/exceptions/metadata.h>
#include <vital/exceptions/video.h>
#include <vital/util/tokenize.h>
#include <vital/klv/convert_metadata.h>
#include <vital/klv/misp_time.h>
#include <vital/klv/klv_data.h>

#include <arrows/vxl/image_container.h>

#include <vidl/vidl_config.h>
#include <vidl/vidl_ffmpeg_istream.h>
#include <vidl/vidl_convert.h>

#include <kwiversys/SystemTools.hxx>

#include <mutex>
#include <memory>
#include <vector>
#include <sstream>


namespace kwiver {
namespace arrows {
namespace vxl {

// ------------------------------------------------------------------
// Private implementation class
class vidl_ffmpeg_video_input::priv
{
public:
  /// Constructor
  priv()
    : c_start_at_frame( 0 ),
      c_stop_after_frame( 0 ),
      c_frame_skip( 1 ),
      c_time_source( "none" ), // initialization string
      c_time_scan_frame_limit( 100 ),
      c_use_metadata( true ),
      d_have_frame( false ),
      d_at_eov( false ),
      d_frame_advanced( false ),
      d_num_frames( 0 ),
      d_have_frame_time( false ),
      d_have_abs_frame_time( false ),
      d_have_metadata( false ),
      d_is_seekable( false ),
      d_have_loop_vars( false ),
      pts_of_meta_ts( 0.0 ),
      meta_ts( 0 ),
      d_frame_time( 0 ),
      d_frame_number( 1 ),
      d_frame_number_offset(0)
  { }


  vidl_ffmpeg_istream d_video_stream;
  vital::logger_handle_t d_logger; // for logging in priv methods

  // Configuration values
  unsigned int c_start_at_frame;
  unsigned int c_stop_after_frame;
  unsigned int c_frame_skip;
  std::string  c_time_source; // default sources string
  std::vector< std::string >  c_time_source_list;
  int c_time_scan_frame_limit; // number of frames to scan looking for time

  /**
   * If this is set then we ignore any metadata included in the video stream.
   */
  bool c_use_metadata;

  // local state
  bool d_have_frame;
  bool d_at_eov;
  bool d_frame_advanced;

  /**
   * This holds the number of frames in the video. If it is set to -2 it means
   * this number still needs to be calculated.
   */
  size_t d_num_frames;

  /**
   * Storage for the metadata map.
   */
  vital::metadata_map::map_metadata_t d_metadata_map;

  /**
   * This is set to indicate that we can supply a frame time of some
   * form. If this is false, the output timestamp will not have a time
   * set. This also is used to report the HAS_FRAME_TIME capability.
   */
  bool d_have_frame_time;

  /**
   * This is set to indicate that we can supply an absolute frame time
   * rather than a relative frame time. This value is used to report
   * the HAS_ABSOLUTE_FRAME_TIME capability.
   */
  bool d_have_abs_frame_time;

  /**
   * This is set to indicate we can supply video metadata and is used
   * to report the HAS_METADATA capability.
   */
  bool d_have_metadata;

  /**
   * This is set to indicate the video stream is seekable by frame and is used
   * to report the IS_SEEKABLE capability.
   */
  bool d_is_seekable;

  /**
   * This is set to indicate that any variables that require a pass through the
   * video like the number of frames or the metadata map have already been
   * determined.
   */
  bool d_have_loop_vars;

  double pts_of_meta_ts;            // probably seconds
  vital::time_us_t meta_ts; // time in usec

  // used to create timestamp output
  vital::time_us_t d_frame_time; // usec
  vital::frame_id_t d_frame_number;

  // frames to add or subtract to make first frame number == 1.
  vital::timestamp::frame_t d_frame_number_offset;

  std::string video_path; // name of video we opened

  std::deque<uint8_t> md_buffer; // working buffer for metadata stream
  kwiver::vital::metadata_vector metadata_collection; // current collection

  kwiver::vital::convert_metadata converter; // metadata converter object

  static std::mutex s_open_mutex;


  // ==================================================================
  /*
   * @brief Process metadata byte stream.
   *
   * This method adds the supplied bytes to the metadata buffer and
   * then tests to see if we have collected enough bytes to make a
   * full metadata packet. If not, then we just return, leaving any
   * current metadata as it was.
   *
   * If a complete klv packet has been received, it is processed and
   * the existing metadata collection is added to the current list of
   * metadata packets.
   *
   * @param curr_md Stream of metadata bytes.
   *
   * @return \b true if there was enough metadata to process.
   */
  bool process_metadata( std::deque<uint8_t> const& curr_md )
  {
    bool retval(false);

    // Add new metadata to the end of current metadata stream
    md_buffer.insert(md_buffer.end(), curr_md.begin(), curr_md.end());
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
        LOG_WARN( this->d_logger, "Metadata exception: " << e.what() );
        continue;
      }

      // If the metadata was even partially decided, then add to the list.
      if ( ! meta->empty() )
      {
        kwiver::vital::timestamp ts;
        ts.set_frame( this->d_frame_number );

        if ( this->d_have_frame_time )
        {
          ts.set_time_usec( this->d_frame_time );
        }

        meta->set_timestamp( ts );

        meta->add( NEW_METADATA_ITEM( vital::VITAL_META_VIDEO_FILENAME,
                                      video_path ) );
        this->metadata_collection.push_back( meta );

        // indicate we have found
        retval = true;
      } // end valid metadata packet.
    } // end while

    // if no metadata from the stream, add a basic metadata item
    // containing video name and timestamp
    if (this->metadata_collection.empty())
    {
      auto meta = std::make_shared<kwiver::vital::metadata>();
      kwiver::vital::timestamp ts;
      ts.set_frame(this->d_frame_number);

      if (this->d_have_frame_time)
      {
        ts.set_time_usec(this->d_frame_time);
      }

      meta->set_timestamp(ts);

      meta->add(NEW_METADATA_ITEM(vital::VITAL_META_VIDEO_FILENAME,
        video_path));

      this->metadata_collection.push_back(meta);

      // TODO decide if this function should return true only for KLV metadata
      retval = true;
    }

    return retval;
  }


 // ------------------------------------------------------------------
  /*
   * @brief Initialize timestamp for video.
   *
   * This method initializes the timestamp at the start of a video,
   * since we need a timestamp for the first frame. It scans ahead in
   * the input stream until it gets a time marker of the specified
   * type.
   *
   * @return \b true if timestamp has been determined.
   */
  bool init_timestamp( std::string  time_source )
  {
    bool retval( true );

    meta_ts = 0.0;
    if ( ! this->d_video_stream.advance() )
    {
      return false;
    }

    // Determine which option has been selected to generate frame time;
    if ( time_source == "misp" )
    {
      retval = misp_time();
    }
    else if ( time_source == "klv0601" )
    {
      retval = klv_time( kwiver::vital::convert_metadata::MISB_0601 );
    }
    else if ( time_source == "klv0104" )
    {
      retval = klv_time( kwiver::vital::convert_metadata::MISB_0104 );
    }
    else if ( time_source == "none" )
    {
      d_have_frame_time = false;
      return true;              // optimized return
    }
    else
    {
      std::stringstream str;
      str <<  "Unknown time source specified \"" << time_source << "\".";
      throw kwiver::vital::video_config_exception( str.str() );
    }

    // If we have located a start time in the video, save the PTS for
    // that point in the video. The video should be left positioned
    // where the time was located. We will get the PTS of a frame and,
    // using this pts_of_meta_ts, be able to adjust the time we got
    // from the metadata correctly.
    if (meta_ts != 0)
    {
      pts_of_meta_ts = d_video_stream.current_pts();
      d_frame_time = meta_ts;
    }

    // Tried seeking to the beginning but some videos don't
    // want to seek, even to the start.  So reload the video.
    {
      std::lock_guard< std::mutex > lock( s_open_mutex );
      d_video_stream.open( this->video_path ); // Calls close on current video
    }

    if ( ! d_video_stream.advance() )
    {
      retval = false;
    }

    // if, after advancing, the PTS is still zero, then we can not
    // establish a relative time reference
    d_have_frame_time = ( 0 != d_video_stream.current_pts() );

    // Clear any old metadata
    metadata_collection.clear();

    return retval;
  } // init_timestamp

// ------------------------------------------------------------------
  bool misp_time()
  {
    int frame_count( c_time_scan_frame_limit );
    bool retval(false);
    int64_t ts = 0;

    do
    {
      std::vector< unsigned char > pkt_data = d_video_stream.current_packet_data();

      if ( kwiver::vital::find_MISP_microsec_time(  pkt_data, ts ) )
      {
        meta_ts = ts; // in usec
        LOG_DEBUG( this->d_logger, "Found MISP frame time:" << meta_ts );

        d_have_abs_frame_time = true;
        retval = true;
      }
    }
    while ( ( meta_ts == 0.0 )
            && d_video_stream.advance()
            && (( c_time_scan_frame_limit == 0) || frame_count-- ));

    return retval;
  } // misp_time


// ------------------------------------------------------------------
  bool klv_time( std::string type )
  {
    int frame_count( c_time_scan_frame_limit );
    bool retval(false);

    do
    {
      // skip ahead until we get some metadata
      if ( d_video_stream.current_metadata().empty() )
      {
        continue;
      }

      //It might be more accurate to get the second unique timestamp instead of the first
      std::deque< vxl_byte > curr_md = d_video_stream.current_metadata();
      if (process_metadata( curr_md ) )
      {
        // A metadata collection was created
        // check to see if it is of the desired type.
        std::string collection_type;
        for( auto meta : this->metadata_collection)
        {
          // Test to see if the collection is from the specified standard (0104/0601)
          if (meta->has( kwiver::vital::VITAL_META_METADATA_ORIGIN ) )
          {
            collection_type = meta->find( kwiver::vital::VITAL_META_METADATA_ORIGIN ).as_string();

            if (type == collection_type)
            {
              if (meta->has( kwiver::vital::VITAL_META_UNIX_TIMESTAMP ) )
              {
                // Get unix timestamp as usec
                meta_ts = meta->find( kwiver::vital::VITAL_META_UNIX_TIMESTAMP ).as_uint64();

                LOG_DEBUG( this->d_logger, "Found initial " << type << " timestamp: " << meta_ts );

                d_have_abs_frame_time = true;
                retval = true;
              } // has time element
            } // correct metadata type
          } // has metadata origin
        } // foreach over all metadata packets
      } // end if processed metadata collection
    }
    while ( ( meta_ts == 0 )
            && d_video_stream.advance()
            && ( (c_time_scan_frame_limit == 0) || frame_count-- ) );

    return retval;
  } // klv_time

// ------------------------------------------------------------------
  void push_metadata_to_map(vital::timestamp::frame_t fn)
  {
    if (fn >= c_start_at_frame &&
        (c_stop_after_frame == 0 || fn <= c_stop_after_frame) &&
        c_use_metadata)
    {
      metadata_collection.clear();
      auto curr_md = d_video_stream.current_metadata();
      if (process_metadata( curr_md ) )
      {
        std::pair<vital::timestamp::frame_t, vital::metadata_vector>
          el(fn, metadata_collection);
        d_metadata_map.insert( el );
      }
    }
  }

// ------------------------------------------------------------------
  void process_loop_dependencies()
  {
    // is stream open?
    if ( ! d_video_stream.is_open() )
    {
      throw vital::file_not_read_exception( video_path, "Video not open" );
    }

    if ( !d_have_loop_vars )
    {
      if ( d_video_stream.is_seekable() )
      {
        std::lock_guard< std::mutex > lock( s_open_mutex );

        d_num_frames = d_frame_number;

        // Add metadata for current frame
        push_metadata_to_map(d_num_frames);

        // Advance video stream to end
        while ( d_video_stream.advance())
        {
          d_num_frames++;
          if ( (d_num_frames - 1) % c_frame_skip == 0 )
          {
            push_metadata_to_map(d_num_frames);
          }
        }

        metadata_collection.clear();

        // Close and reopen to reset
        d_video_stream.open( video_path );

        // Advance back to original frame number
        unsigned int frame_num = d_video_stream.frame_number()
                               + d_frame_number_offset + 1;
        while ( frame_num < d_frame_number &&
                d_video_stream.advance() )
        {
          ++frame_num;
          if ((frame_num - 1) % c_frame_skip == 0 )
          {
            push_metadata_to_map(frame_num);
          }
        }
      }

      d_have_loop_vars = true;
    }
  }

}; // end of internal class.

// static open interlocking mutex
std::mutex vidl_ffmpeg_video_input::priv::s_open_mutex;


// ==================================================================
vidl_ffmpeg_video_input
::vidl_ffmpeg_video_input()
  : d( new priv() )
{
  attach_logger( "arrows.vxl.video_input" ); // get appropriate logger
  d->d_logger = this->logger();
}


vidl_ffmpeg_video_input
::~vidl_ffmpeg_video_input()
{
  d->d_video_stream.close( );
}


// ------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
vidl_ffmpeg_video_input
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "time_scan_frame_limit", d->c_time_scan_frame_limit,
                     "Number of frames to be scanned searching input video for embedded time. "
                      "If the value is zero, the whole video will be scanned." );

  config->set_value( "start_at_frame", d->c_start_at_frame,
                     "Frame number (from 1) to start processing video input. "
                     "If set to zero, start at the beginning of the video." );

  config->set_value( "stop_after_frame", d->c_stop_after_frame,
                     "Number of frames to supply. If set to zero then supply all frames after start frame." );

  config->set_value( "output_nth_frame", d->c_frame_skip,
                     "Only outputs every nth frame of the video starting at the first frame. The output "
                     "of num_frames still reports the total frames in the video but skip_frame is valid "
                     "every nth frame only and there are metadata_map entries for only every nth frame.");

  config->set_value( "use_metadata", d->c_use_metadata,
                     "Whether to use any metadata provided by the for video stream." );

  config->set_value( "absolute_time_source", d->c_time_source,
                     "List of sources for absolute frame time information. "
                     "This entry specifies a comma separated list of sources that are "
                     "tried in order until a valid time source is found. "
                     "If an absolute time source is found, it is used in the output time stamp. "
                     "Absolute times are derived from the metadata in the video stream. "
                     "Valid source names are \"none\", \"misp\", \"klv0601\", \"klv0104\".\n"
                     "Where:\n"
                     "    none - do not supply absolute time\n"
                     "    misp - use frame embedded time stamps.\n"
                     "    klv0601 - use klv 0601 format metadata for frame time\n"
                     "    klv0104 - use klv 0104 format metadata for frame time\n"
                     "Note that when \"none\" is found in the list no further time sources will be evaluated, "
                     "the output timestamp will be marked as invalid, and the HAS_ABSOLUTE_FRAME_TIME capability "
                     "will be set to false.  The same behavior occurs when all specified sources are tried and "
                     "no valid time source is found."
    );

  return config;
}


// ------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
vidl_ffmpeg_video_input
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.

  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d->c_start_at_frame = config->get_value<vital::frame_id_t>(
    "start_at_frame", d->c_start_at_frame );

  d->c_stop_after_frame = config->get_value<vital::frame_id_t>(
    "stop_after_frame", d->c_stop_after_frame );

  d->c_frame_skip = config->get_value<vital::timestamp::frame_t>(
    "output_nth_frame", d->c_frame_skip );

  kwiver::vital::tokenize( config->get_value<std::string>( "time_source", d->c_time_source ),
            d->c_time_source_list, " ,", kwiver::vital::TokenizeTrimEmpty );
}


// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::check_configuration(vital::config_block_sptr config) const
{
  bool retcode(true); // assume success

  // validate time source
  bool valid_src( true );
  std::vector< std::string > time_source;
  kwiver::vital::tokenize( config->get_value<std::string>( "time_source", d->c_time_source ),
            time_source, " ,", kwiver::vital::TokenizeTrimEmpty );

  for( auto source : time_source )
  {
    if (source != "none"
        && source != "misp"
        && source != "klv0601"
        && source != "klv0104")
    {
      valid_src = false;
      break;
    }
  }

  if ( ! valid_src )
  {
    LOG_ERROR( logger(), "time source must be a comma separated list of one or more "
               "of the following strings: \"none\", \"misp\", \"klv0601\", \"klv0104\"" );
    retcode = false;
  }

  // validate start frame
  if (config->has_value("start_at_frame"))
  {
    vital::frame_id_t frame = config->get_value<vital::frame_id_t>("start_at_frame");
    //  zero indicates not set, otherwise must be 1 or greater
    if (frame < 0)
    {
      LOG_ERROR(logger(), "start_at_frame must be greater than 0");
      retcode = false;
    }
  }

  return retcode;
}


// ------------------------------------------------------------------
void
vidl_ffmpeg_video_input
::open( std::string video_name )
{
#if ! VIDL_HAS_FFMPEG
  throw kwiver::vital::video_config_exception( "vidl ffmpeg support is not available from VXL. "
                                               "Rebuild VXL with ffmpeg support." );
#endif

  this->close(); // close video stream and reset internal state

  d->video_path = video_name;

  // If the open succeeds, it will already have read the first frame.
  // avcodec_open2 which is called by open is not thread safe so we need to lock.
  {
    std::lock_guard< std::mutex > lock( d->s_open_mutex );

    if ( ! kwiversys::SystemTools::FileExists( video_name ) )
    {
      // Throw exception
      throw kwiver::vital::file_not_found_exception( video_name, "File not found" );
    }

    if( ! d->d_video_stream.open( video_name ) )
    {
      throw kwiver::vital::video_runtime_exception( "Video stream open failed for unknown reasons");
    }
  }

  d->d_at_eov = false;
  d->d_frame_advanced = false;
  d->d_frame_number = 1;

  // check for metadata
  d->d_have_metadata = d->d_video_stream.has_metadata();

  // check for seekability
  d->d_is_seekable = d->d_video_stream.is_seekable();

  // We already have required frame
  // See if we can generate a time base
  d->d_have_frame = true;
  bool time_found( false );
  for( auto time_source : d->c_time_source_list )
  {
    LOG_DEBUG( d->d_logger, "Looking for " << time_source << " as time source" );
    if( d->init_timestamp( time_source ) )  // will call advance()
    {
      LOG_DEBUG( d->d_logger, "Found " << time_source << " as time source" );
      time_found = true;
      break;
    }
  }

  // the initial frame should be 0, but sometimes it is not, so we capture the initial
  // frame as an offset
  d->d_frame_number_offset = - static_cast<int>(d->d_video_stream.frame_number());

  if ( ! time_found )
  {
    LOG_ERROR( logger(), "Failed to initialize the timestamp for: " << d->video_path );
    throw kwiver::vital::video_stream_exception( "could not initialize timestamp" );
  }

  // Move stream to starting frame if needed
  if ( d->c_start_at_frame != 0 && d->c_start_at_frame > 1 )
  {
    // move stream to specified frame number
    unsigned int frame_num = d->d_video_stream.frame_number()
                           + d->d_frame_number_offset + 1;

    while (frame_num < d->c_start_at_frame)
    {
      if( ! d->d_video_stream.advance() )
      {
        break;
      }

      frame_num = d->d_video_stream.frame_number()
                + d->d_frame_number_offset + 1;
    }

    d->d_frame_number = frame_num;
  }

  // Set capabilities
  set_capability(vital::algo::video_input::HAS_TIMEOUT, false );

  set_capability(vital::algo::video_input::HAS_EOV, true );
  set_capability(vital::algo::video_input::HAS_FRAME_DATA, true);
  set_capability(vital::algo::video_input::HAS_FRAME_NUMBERS, true );
  set_capability(vital::algo::video_input::HAS_FRAME_TIME, d->d_have_frame_time  );
  set_capability(vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME,
                 (d->d_have_frame_time & d->d_have_abs_frame_time) );
  set_capability(vital::algo::video_input::HAS_METADATA, d->d_have_metadata  );
  set_capability(vital::algo::video_input::IS_SEEKABLE, d->d_is_seekable );
}


// ------------------------------------------------------------------
void
vidl_ffmpeg_video_input
::close()
{
  d->d_video_stream.close();

  d->d_have_frame = false;
  d->d_at_eov = false;
  d->d_frame_advanced = false;
  d->d_num_frames = 0;
  d->d_have_frame_time = false;
  d->d_have_abs_frame_time = false;
  d->d_have_metadata = false;
  d->d_is_seekable = false;
  d->d_have_loop_vars = false;
  d->d_frame_time = 0;
  d->d_frame_number = 1;
}


// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::next_frame( kwiver::vital::timestamp& ts,
              uint32_t timeout )
{
  if (d->d_at_eov)
  {
    return false;
  }

  // is stream open?
  if ( ! d->d_video_stream.is_open() )
  {
    throw vital::file_not_read_exception( d->video_path, "Video not open" );
  }

  // Sometimes we already have the frame available.
  if ( d->d_have_frame )
  {
    d->d_have_frame = false;
  }
  else
  {
    do
    {
      if( ! d->d_video_stream.advance() )
      {
        d->d_at_eov = true;
        return false;
      }
      d->d_frame_number = d->d_video_stream.frame_number()
                        + d->d_frame_number_offset + 1;
    } while ( (d->d_frame_number - 1) % d->c_frame_skip != 0 );
  }

  // ---- Calculate time stamp ----
  // Metadata packets may not exist for each frame, so use the diff in
  // presentation time stamps to foward the first metadata time stamp.
  double pts_diff = ( d->d_video_stream.current_pts() - d->pts_of_meta_ts ) * 1e6;
  d->d_frame_time = d->meta_ts + pts_diff;


  ts = this->frame_timestamp();

  if( (d->c_stop_after_frame != 0) && (ts.get_frame() > d->c_stop_after_frame))
  {
    d->d_at_eov = true;  // logical end of file
    return false;
  }

  d->d_frame_advanced = true;

  return true;
}

// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  // is stream open?
  if ( ! d->d_video_stream.is_open() )
  {
    throw vital::file_not_read_exception( d->video_path, "Video not open" );
  }

  // negative or zero frame number not allowed
  if ( frame_number <= 0 )
  {
    return false;
  }

  // Check if requested frame would have been skipped
  if ( ( frame_number - 1 )%d->c_frame_skip != 0 )
  {
    return false;
  }

  // Check if requested frame is valid
  if ( (d->c_stop_after_frame != 0 && d->c_stop_after_frame < frame_number )
        || frame_number < d->c_start_at_frame )
  {
    return false;
  }

  int curr_frame_num = d->d_video_stream.frame_number()
                     + d->d_frame_number_offset + 1;

  // If current frame number is greater than requested frame reopen
  // file to reset to start
  if (curr_frame_num > frame_number)
  {
    std::lock_guard< std::mutex > lock( d->s_open_mutex );
    d->d_video_stream.open( d->video_path ); // Calls close on current video
    curr_frame_num = d->d_video_stream.frame_number()
                   + d->d_frame_number_offset + 1;
  }

  // Just advance video until the requested frame is reached
  for (int i=curr_frame_num; i<frame_number; ++i)
  {
    if( ! d->d_video_stream.advance() )
    {
      d->d_at_eov = true;
      return false;
    }
    else
    {
      d->d_frame_advanced = true;
      d->d_have_frame = false;
    }
  }
  // if seeking to the first frame we need to mark the video as having advanced
  // otherwise we will get the first frame twice.
  if (frame_number == 1)
  {
    d->d_frame_advanced = true;
    d->d_have_frame = false;
  }

  // ---- Calculate time stamp ----
  // Metadata packets may not exist for each frame, so use the diff in
  // presentation time stamps to foward the first metadata time stamp.
  double pts_diff = ( d->d_video_stream.current_pts() - d->pts_of_meta_ts ) * 1e6;
  d->d_frame_time = d->meta_ts + pts_diff;
  d->d_frame_number = d->d_video_stream.frame_number()
                    + d->d_frame_number_offset + 1;


  ts = this->frame_timestamp();

  // ---- process metadata ---
  d->metadata_collection.clear(); // erase old metadata packets

  return true;
}


// ------------------------------------------------------------------
kwiver::vital::timestamp
vidl_ffmpeg_video_input
::frame_timestamp() const
{
  if (d->d_at_eov)
  {
    return {};
  }

  // We don't always have all components of a timestamp, so start with
  // an invalid TS and add the data we have.
  kwiver::vital::timestamp ts;
  ts.set_frame( d->d_frame_number );

  return ts;
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
vidl_ffmpeg_video_input
::frame_image( )
{
  if (d->d_at_eov)
  {
    return kwiver::vital::image_container_sptr();
  }

  // We succeed in the step if we can convert the frame to RGB.
  vil_image_view<vxl_byte> img;
  vidl_frame_sptr vidl_frame = d->d_video_stream.current_frame();
  bool result = vidl_convert_to_view( *vidl_frame,
                                      img,
                                      VIDL_PIXEL_COLOR_RGB );

  if ( ! result )
  {
    throw kwiver::vital::video_stream_exception( "could not convert image to vidl format" );
  }

  // make an image container and add the first metadata object, if there is one
  auto img_cont = std::make_shared<vxl::image_container>(img);
  auto mdv = this->frame_metadata();
  if (!mdv.empty())
  {
    img_cont->set_metadata(mdv[0]);
  }

  return img_cont;
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
vidl_ffmpeg_video_input
::frame_metadata()
{
  if (d->d_at_eov)
  {
    return kwiver::vital::metadata_vector();
  }

  // ---- process metadata ---
  // If the vector is empty, then try to convert metadata.
  if ( d->metadata_collection.empty() )
  {
    // will manage metadata collection object.
    d->process_metadata(d->d_video_stream.current_metadata());
  }

  return d->metadata_collection;
}


kwiver::vital::metadata_map_sptr
vidl_ffmpeg_video_input
::metadata_map()
{
  d->process_loop_dependencies();

  return std::make_shared<kwiver::vital::simple_metadata_map>(d->d_metadata_map);
}


// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::end_of_video() const
{
  return d->d_at_eov;
}


// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::good() const
{
  return d->d_video_stream.is_valid() && d->d_frame_advanced;
}

// ------------------------------------------------------------------
bool
vidl_ffmpeg_video_input
::seekable() const
{
  return d->d_video_stream.is_seekable();
}

// ------------------------------------------------------------------
size_t
vidl_ffmpeg_video_input
::num_frames() const
{
  // Const cast needed so this can be called from const method
  auto privateData = const_cast<vidl_ffmpeg_video_input::priv*>((d.get()));
  privateData->process_loop_dependencies();

  if (d->c_stop_after_frame > 0)
  {
    return std::min
        (static_cast<size_t>(d->c_stop_after_frame + 1), d->d_num_frames)
        - d->c_start_at_frame;
  }
  else
  {
    return d->d_num_frames - d->c_start_at_frame;
  }
}

} } } // end namespace
