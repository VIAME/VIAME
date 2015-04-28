/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include "frame_list_process.h"

#include <kwiver/pipeline_types/kwiver.h>
#include <kwiver/timestamp.h>

#include <maptk/modules.h>
#include <kwiver/image_container.h>
#include <kwiver/image.h>
#include <maptk/algo/image_io.h>
#include <kwiver/exceptions.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/datum.h>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include <vector>
#include <stdint.h>
#include <fstream>

// -- DEBUG
#if defined DEBUG
#include <maptk/ocv/image_container.h>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

namespace bfs = boost::filesystem;
namespace algo = maptk::algo;

namespace kwiver
{

  // (config-key, value-type, default-value, description )
  create_config_trait( image_list_file, std::string, "", "Name of file that contains list of image file names." );
  create_config_trait( image_reader, std::string, "", "Image reader type. Must be \"ocv\" or \"vxl\"" );
  create_config_trait( frame_time, double, "0.3333333", "Inter frame time in seconds" );

//----------------------------------------------------------------
// Private implementation class
class frame_list_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_config_image_list_filename;
  std::string m_config_image_reader;
  double m_config_frame_time;

  // process local data
  std::vector < kwiver::path_t > m_files;
  std::vector < kwiver::path_t >::const_iterator m_current_file;
  timestamp::frame_t m_frame_number;
  timestamp::time_t m_frame_time;

  // processing classes
  algo::image_io_sptr m_image_reader;

}; // end priv class


// ================================================================

frame_list_process
::frame_list_process( sprokit::config_t const& config )
  : process( config ),
    d( new frame_list_process::priv )
{
  maptk::register_modules();
  make_ports();
  make_config();
}


frame_list_process
::~frame_list_process()
{
}


// ----------------------------------------------------------------
void frame_list_process
::_configure()
{

  // Examine the configuration
  d->m_config_image_list_filename = config_value_using_trait( image_list_file );
  d->m_config_image_reader        = config_value_using_trait( image_reader );
  d->m_config_frame_time          = config_value_using_trait( frame_time );

  // Convert sprokit config to maptk config for algorithms
  sprokit::config_t proc_config = get_config(); // config for process
  maptk::config_block_sptr algo_config = maptk::config_block::empty_config();

  //+ convert_config( proc_config, algo_config );

  // instantiate image reader and converter based on config type
  algo::image_io::set_nested_algo_configuration( "image_reader", algo_config, d->m_image_reader);
  if (0 == d->m_image_reader )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Error configuring \"feature_tracker\". Unable to create image reader." );
  }

  sprokit::process::_configure();
}


// ----------------------------------------------------------------
// Post connection initialization
void frame_list_process
::_init()
{
  // open file and read lines
  std::ifstream ifs( d->m_config_image_list_filename.c_str() );
  if ( ! ifs )
  {
    std::stringstream msg;
    msg <<  "Could not open image list \"" << d->m_config_image_list_filename << "\"";
    throw sprokit::invalid_configuration_exception( this->name(), msg.str() );
  }

  // verify and get file names in a list
  for ( std::string line; std::getline( ifs, line ); )
  {
    d->m_files.push_back( line );
    if ( ! bfs::exists( d->m_files.back() ) )
    {
      throw maptk::path_not_exists( d->m_files.back() );
    }
  } // end for

  d->m_current_file = d->m_files.begin();
  d->m_frame_number = 0;

  process::_init();
}


// ----------------------------------------------------------------
void frame_list_process
::_step()
{

  if ( d->m_current_file != d->m_files.end() )
  {
    // still have an image to read
    std::string a_file = d->m_current_file->string();

    // \todo add log message
    std::cerr << "DEBUG - reading image from file \"" << a_file << "\"\n";

    // read image file
    //
    // This call returns a *new* image container. This is good since
    // we are going to pass it downstream using the sptr.
    kwiver::image_container_sptr img;
    img = d->m_image_reader->load( a_file );

    // --- debug
#if defined DEBUG
    cv::Mat image = maptk::ocv::image_container::maptk_to_ocv( img->get_image() );
    namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                 // Wait for a keystroke in the window
#endif
    // -- end debug

    // update timestamp
    ++d->m_frame_number;
    d->m_frame_time += d->m_config_frame_time;

    kwiver::timestamp frame_ts( d->m_frame_time, d->m_frame_number );

    push_to_port_using_trait( timestamp, frame_ts );
    push_to_port_using_trait( image, img );

    ++d->m_current_file;
  }
  else
  {
    // \todo log message
    std::cerr << "DEBUG - end of input reached, process terminating\n";

    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( timestamp, dat );
    push_datum_to_port_using_trait( image, dat );
  }

  sprokit::process::_step();
}


// ----------------------------------------------------------------
void frame_list_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_output_port_using_trait( timestamp, required );
  declare_output_port_using_trait( image, required );
}


// ----------------------------------------------------------------
void frame_list_process
::make_config()
{
  declare_config_using_trait( image_list_file );
  declare_config_using_trait( image_reader );
  declare_config_using_trait( frame_time );
}


// ================================================================
frame_list_process::priv
::priv()
  :m_frame_number(1),
   m_frame_time(0)
{
}


frame_list_process::priv
::~priv()
{
}

} // end namespace
