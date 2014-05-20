/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "frame_list_process.h"

#include <types/maptk.h>
#include <types/kwiver.h>
#include <core/exceptions.h>
#include <core/timestamp.h>

#include <maptk/core/image_container.h>
#include <maptk/core/image.h>
#include <maptk/core/algo/image_io.h>
#include <maptk/core/exceptions.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/datum.h>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include <vector>
#include <stdint.h>
#include <fstream>

namespace bfs = boost::filesystem;
namespace algo = maptk::algo;

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class frame_list_process::priv
{
public:
  priv();
  ~priv();


  static sprokit::process::port_t const port_timestamp;
  static sprokit::process::port_t const port_image;

  // Configuration values
  std::string m_config_image_list_filename;
  static sprokit::config::key_t const config_image_list_filename;
  static sprokit::config::value_t const default_image_list_filename;

  std::string m_config_image_reader;
  static sprokit::config::key_t const config_image_reader;
  static sprokit::config::value_t const default_image_reader;

  double m_config_frame_time;
  static sprokit::config::key_t const config_frame_time;
  static sprokit::config::value_t const default_frame_time;

  // process local data
  std::vector < maptk::path_t > m_files;
  std::vector < maptk::path_t >::const_iterator m_current_file;
  timestamp::frame_t m_frame_number;
  timestamp::time_t m_frame_time;

  // processing classes
  algo::image_io_sptr m_image_reader;

}; // end priv class

// -- ports --
sprokit::process::port_t const port_timestamp = sprokit::process::port_t("timestamp");
sprokit::process::port_t const port_image = sprokit::process::port_t("image");

// -- config --
sprokit::config::key_t const config_image_list_filenamey = sprokit::config::key_t( "output_directory" );
sprokit::config::value_t const default_image_list_filename = sprokit::config::value_t( "" );

sprokit::config::key_t const config_image_reader = sprokit::config::key_t( "image_reader_type" );
sprokit::config::value_t const default_image_reader = sprokit::config::value_t( "" );

sprokit::config::key_t const config_frame_time = sprokit::config::key_t( "frame_time" );
sprokit::config::value_t const default_frame_time = sprokit::config::value_t( "0.03333333333" ); // 30 Hz


// ================================================================

frame_list_process
::frame_list_process( sprokit::config_t const& config )
  : process( config ),
    d( new frame_list_process::priv )
{
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
  d->m_config_image_list_filename = config_value< std::string > ( frame_list_process::priv::config_image_list_filename );
  d->m_config_image_reader        = config_value< std::string > ( frame_list_process::priv::config_image_reader );
  d->m_config_frame_time          = config_value< double > ( frame_list_process::priv::config_frame_time );

  //+ a better approach is to get the config from this process using get_config()
  // and pass it to the maptk algorithm

  // Create default maptk config block
  maptk::config_block_sptr config = maptk::config_block::empty_config("frame_list_process");


  // add to config block
  config->set_value( "image_reader:type", d->m_config_image_reader );

  // instantiate image reader and converter based on config type
  algo::image_io::set_nested_algo_configuration( "image_reader", config, d->m_image_reader);

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
  }

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
    ++d->m_current_file;

    // still have an image to read
    std::string a_file = d->m_current_file->string();

    // read image file
    //
    // This call returns a *new* image container. This is good since
    // we are going to pass it downstream using the sptr.
    maptk::image_container_sptr img = d->m_image_reader->load( a_file );

    // update timestamp
    ++d->m_frame_number;
    d->m_frame_time += d->m_config_frame_time;

    kwiver::timestamp frame_ts( d->m_frame_time, d->m_frame_number );

    push_to_port_as< kwiver::timestamp > ( priv::port_timestamp, frame_ts );
    push_to_port_as< maptk::image_container_sptr > ( priv::port_image, img );
  }
  else
  {
    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port( priv::port_timestamp, dat );
    push_datum_to_port( priv::port_image, dat );
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

  declare_output_port(
    priv::port_timestamp,
    kwiver_timestamp,
    required,
    port_description_t( "Timestamp for input image." ) );

  declare_output_port(
    priv::port_image,
    maptk_image_container,
    required,
    port_description_t( "Single frame image." ) );
}


// ----------------------------------------------------------------
void frame_list_process
::make_config()
{
  declare_configuration_key(
    priv::config_image_list_filename,
    priv::default_image_list_filename,
    sprokit::config::description_t( "Name of file that contains list of image file names." ));

  declare_configuration_key(
    priv::config_image_reader,
    priv::default_image_reader,
    sprokit::config::description_t( "Image reader type. Must be \"ocv\" or \"vxl\"" ));

  declare_configuration_key(
    priv::config_frame_time,
    priv::default_frame_time,
    sprokit::config::description_t( "Inter frame time in seconds" ));
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
