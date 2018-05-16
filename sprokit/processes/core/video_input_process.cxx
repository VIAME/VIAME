/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include "video_input_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>

#include <vital/algo/video_input.h>
#include <vital/exceptions.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/datum.h>

// -- DEBUG
#if defined DEBUG
#include <arrows/algorithms/ocv/image_container.h>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

namespace algo = kwiver::vital::algo;

namespace kwiver {

//                 (config-key, value-type, default-value, description )
create_config_trait( video_reader, std::string, "", "Name of video input algorithm. "
  " Name of the video reader algorithm plugin is specified as video_reader:type = <algo-name>" );
create_config_trait( video_filename, std::string, "", "Name of video file." );
create_config_trait( frame_time, double, "0.03333333",
                     "Inter frame time in seconds. "
                     "If the input video stream does not supply frame times, "
                     "this value is used to create a default timestamp. "
                     "If the video stream has frame times, then those are used." );

//----------------------------------------------------------------
// Private implementation class
class video_input_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string                           m_config_video_filename;
  kwiver::vital::time_us_t              m_config_frame_time;
  bool                                  m_has_config_frame_time;

  kwiver::vital::algo::video_input_sptr m_video_reader;
  kwiver::vital::algorithm_capabilities m_video_traits;

  kwiver::vital::frame_id_t             m_frame_number;
  kwiver::vital::time_us_t              m_frame_time;

  kwiver::vital::metadata_vector        m_last_metadata;

}; // end priv class


// ================================================================

video_input_process
::video_input_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new video_input_process::priv )
{
  make_ports();
  make_config();
}


video_input_process
::~video_input_process()
{
}


// ----------------------------------------------------------------
void video_input_process
::_configure()
{
  scoped_configure_instrumentation();

  // Examine the configuration
  d->m_config_video_filename = config_value_using_trait( video_filename );
  d->m_config_frame_time = static_cast<vital::time_us_t>(
                               config_value_using_trait( frame_time ) * 1e6); // in usec

  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process
  if( algo_config->has_value( "frame_time" ) )
  {
    d->m_has_config_frame_time = true;
  }

  if ( ! algo::video_input::check_nested_algo_configuration( "video_reader", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  // instantiate requested/configured algo type
  algo::video_input::set_nested_algo_configuration( "video_reader", algo_config, d->m_video_reader );
  if ( ! d->m_video_reader )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create video_reader." );
  }
}


// ----------------------------------------------------------------
// Post connection initialization
void video_input_process
::_init()
{
  scoped_init_instrumentation();

  // instantiate a video reader
  d->m_video_reader->open( d->m_config_video_filename ); // throws

  d->m_video_traits = d->m_video_reader->get_implementation_capabilities();
}


// ----------------------------------------------------------------
void video_input_process
::_step()
{
  kwiver::vital::timestamp ts;

  if ( d->m_video_reader->next_frame( ts ) )
  {
    kwiver::vital::metadata_vector metadata;
    kwiver::vital::image_container_sptr frame;
    {
      scoped_step_instrumentation();

      frame = d->m_video_reader->frame_image();

      // --- debug
#if defined DEBUG
      cv::Mat image = algorithms::ocv::image_container::vital_to_ocv( frame->get_image() );
      namedWindow( "Display window", cv::WINDOW_NORMAL ); // Create a window for display.
      imshow( "Display window", image ); // Show our image inside it.

      waitKey(0);                 // Wait for a keystroke in the window
#endif
      // -- end debug

      // update timestamp
      //
      // Sometimes the video source can not determine either the frame
      // number or frame time or both.
      if ( ! d->m_video_traits.capability( kwiver::vital::algo::video_input::HAS_FRAME_DATA ) )
      {
        throw sprokit::invalid_configuration_exception( name(),
                                                        "Video reader selected does not supply image data." );
      }


      if ( d->m_video_traits.capability( kwiver::vital::algo::video_input::HAS_FRAME_NUMBERS ) )
      {
        d->m_frame_number = ts.get_frame();
      }
      else
      {
        ++d->m_frame_number;
        ts.set_frame( d->m_frame_number );
      }

      if ( ! d->m_video_traits.capability( kwiver::vital::algo::video_input::HAS_FRAME_TIME ) )
      {
        // create an internal time standard
        double frame_rate = d->m_video_reader->frame_rate();
        if( ! d->m_video_traits.capability( kwiver::vital::algo::video_input::HAS_FRAME_RATE ) ||
            frame_rate <= 0.0 || d->m_has_config_frame_time )
        {
          d->m_frame_time = d->m_frame_number * d->m_config_frame_time;
        }
        else
        {
          time_t frame_time_usec = ( 1.0 / frame_rate ) * 1e6;
          d->m_frame_time = d->m_frame_number * frame_time_usec;
        }
        ts.set_time_usec( d->m_frame_time );
      }

      // If this reader/video does not have any metadata, we will just
      // return an empty vector.  That is all handled by the algorithm
      // implementation.
      metadata = d->m_video_reader->frame_metadata();

      // Since we want to try to always return valid metadata for this
      // frame - if the returned metadata is empty, then use the last
      // one we received.  The requirement is to always provide the best
      // metadata for a frame. Since metadata appears less frequently
      // than the frames, the metadata returned can be a little old, but
      // it is still the best we have.
      if ( metadata.empty() )
      {
        // The saved one could be empty, but it is the bewt we have.
        metadata = d->m_last_metadata;
      }
      else
      {
        // Now that we have new metadata save it in case we need it later.
        d->m_last_metadata = metadata;
      }
    }

    if( ts.get_frame() < 4294967000 && ts.get_frame() > 0 )
    {
      push_to_port_using_trait( timestamp, ts );
      push_to_port_using_trait( image, frame );
      push_to_port_using_trait( metadata, metadata );
      push_to_port_using_trait( frame_rate, d->m_video_reader->frame_rate() );
    }
  }
  else
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );

    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( timestamp, dat );
    push_datum_to_port_using_trait( image, dat );
    push_datum_to_port_using_trait( metadata, dat );
    push_datum_to_port_using_trait( frame_rate, dat );
  }
}


// ----------------------------------------------------------------
void video_input_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( image, optional );
  declare_output_port_using_trait( metadata, optional );
  declare_output_port_using_trait( frame_rate, optional );
}


// ----------------------------------------------------------------
void video_input_process
::make_config()
{
  declare_config_using_trait( video_reader );
  declare_config_using_trait( video_filename );
  declare_config_using_trait( frame_time );
}


// ================================================================
video_input_process::priv
::priv()
  : m_has_config_frame_time( false ),
    m_frame_number( 1 ),
    m_frame_time( 0 )
{
}


video_input_process::priv
::~priv()
{
}

} // end namespace
