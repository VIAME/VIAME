/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * \file
 * \brief Implementation of the image_file_reader_process
 */

#include "image_file_reader_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/algo/image_io.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>
#include <vital/util/enum_converter.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/datum.h>

#include <kwiversys/SystemTools.hxx>

#include <vector>
#include <stdint.h>
#include <fstream>
#include <string>

// -- DEBUG
#if defined DEBUG
#include <arrows/algorithms/ocv/image_container.h>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( error_mode, std::string, "fail",
                     "How to handle file not found errors. Options are 'abort' and 'skip'. "
                     "Specifying 'fail' will cause an exception to be thrown. "
                     "The 'pass' option will only log a warning and wait for the next file name." );

create_config_trait( path, std::string, "",
                     "Path to search for image file. The format is the same as the standard "
                     "path specification, a set of directories separated by a colon (':')" );

create_config_trait( frame_time, double, "0.3333333", "Inter frame time in seconds. "
                     "The generated timestamps will have the specified number of seconds in the generated "
                     "timestamps for sequential frames. This can be used to simulate a frame rate in a "
                     "video stream application." );

create_config_trait( no_path_in_name, bool, "true",
                     "Set to true if the output image file path should not contain a full path to"
                     "the image file and just contain the file name for the image." );

create_config_trait( image_reader, std::string, "", "Algorithm configuration subblock." )


//----------------------------------------------------------------
// Private implementation class
class image_file_reader_process::priv
{
public:
  priv();
  ~priv();

  enum error_mode_t {           // Error handling modes
    ERROR_ABORT = 1,
    ERROR_SKIP
  };

  // Define the enum converter
  ENUM_CONVERTER( mode_converter, error_mode_t,
                  { "abort",   ERROR_ABORT },
                  { "skip",    ERROR_SKIP }
    )

  // Configuration values
  int m_config_error_mode; // error mode
  std::vector< std::string > m_config_path;
  kwiver::vital::time_us_t m_config_frame_time;
  bool m_no_path_in_name;

  // local state
  kwiver::vital::frame_id_t m_frame_number;
  kwiver::vital::time_us_t m_frame_time;

  // processing classes
  algo::image_io_sptr m_image_reader;

}; // end priv class


// ================================================================

image_file_reader_process
::image_file_reader_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new image_file_reader_process::priv )
{
  make_ports();
  make_config();
}


image_file_reader_process
::~image_file_reader_process()
{
}


// ----------------------------------------------------------------
void image_file_reader_process
::_configure()
{
  scoped_configure_instrumentation();

  // Examine the configuration
  std::string mode = config_value_using_trait( error_mode );
  std::string path = config_value_using_trait( path );
  d->m_config_frame_time = config_value_using_trait( frame_time ) * 1e6; // in usec
  d->m_no_path_in_name = config_value_using_trait( no_path_in_name );

  kwiver::vital::tokenize( path, d->m_config_path, ":", kwiver::vital::TokenizeTrimEmpty );
  d->m_config_path.push_back( "." ); // add current directory

  d->m_config_error_mode = priv::mode_converter().from_string( mode );

  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  algo::image_io::set_nested_algo_configuration( "image_reader", algo_config, d->m_image_reader);
  if ( ! d->m_image_reader )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Unable to create image_reader." );
  }

  algo::image_io::get_nested_algo_configuration( "image_reader", algo_config, d->m_image_reader);

  // instantiate image reader and converter based on config type
  if ( ! algo::image_io::check_nested_algo_configuration( "image_reader", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------
void image_file_reader_process
::_step()
{
  std::string file = grab_from_port_using_trait( image_file_name );

  std::string resolved_file = file;
  if ( ! kwiversys::SystemTools::FileExists( file ) )
  {
    // Resolve against specified path
    std::string resolved_file = kwiversys::SystemTools::FindFile( file, d->m_config_path, true );
    if ( resolved_file.empty() )
    {
      switch (d->m_config_error_mode)
      {
      case priv::ERROR_SKIP:
        LOG_WARN( logger(), "Input file \"" << file << "\" could not be found. Ignoring input." );
        return;

      case priv::ERROR_ABORT:
      default:
        throw kwiver::vital::file_not_found_exception( file, "could not locate file in path" );
      } // end switch
    }
  }

  kwiver::vital::image_container_sptr img_c;
  kwiver::vital::timestamp frame_ts;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "reading image from file \"" << resolved_file << "\"." );

    // read image file
    //
    // This call returns a *new* image container. This is good since
    // we are going to pass it downstream using the sptr.
    img_c = d->m_image_reader->load( resolved_file );

    // --- debug
#if defined DEBUG
    cv::Mat image = algorithms::ocv::image_container::vital_to_ocv( img_c->get_image() );
    namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                 // Wait for a keystroke in the window
#endif
    // -- end debug

    frame_ts = kwiver::vital::timestamp( d->m_frame_time, d->m_frame_number );

    // update timestamp
    ++d->m_frame_number;
    d->m_frame_time += d->m_config_frame_time;
  }

  if ( d->m_no_path_in_name )
  {
    const size_t last_slash_idx = resolved_file.find_last_of("\\/");

    if ( std::string::npos != last_slash_idx )
    {
      resolved_file.erase( 0, last_slash_idx + 1 );
    }
  }

  push_to_port_using_trait( timestamp, frame_ts );
  push_to_port_using_trait( image, img_c );
  push_to_port_using_trait( image_file_name, resolved_file );
}


// ----------------------------------------------------------------
void image_file_reader_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image_file_name, required, "Name of the image file to read. "
                                  "The file is searched for using the specified path in addition to the "
                                  "local directory.");

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( image, optional );
  declare_output_port_using_trait( image_file_name, optional );
}


// ----------------------------------------------------------------
void image_file_reader_process
::make_config()
{
  declare_config_using_trait( frame_time );
  declare_config_using_trait( error_mode );
  declare_config_using_trait( path );
  declare_config_using_trait( image_reader );
  declare_config_using_trait( no_path_in_name );
}


// ================================================================
image_file_reader_process::priv
::priv()
  : m_frame_number( 1 )
  , m_frame_time( 0 )
{
}


image_file_reader_process::priv
::~priv()
{
}

} // end namespace
