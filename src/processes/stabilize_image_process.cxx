/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#include "stabilize_image_process.h"

#include <types/maptk.h>
#include <types/kwiver.h>
#include <core/exceptions.h>
#include <core/timestamp.h>

#include <maptk/core/algo/track_features.h>
#include <maptk/core/algo/compute_ref_homography.h>

#include <maptk/core/image_container.h>
#include <maptk/core/track_set.h>
#include <maptk/core/homography.h>

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class stabilize_image_process::priv
{
public:
  priv();
  ~priv();

  // -- inputs --
  static sprokit::process::port_t const port_timestamp;
  static sprokit::process::port_t const port_image;

  // -- outputs --
  static sprokit::process::port_t const port_homography;


  // Configuration values
  std::string m_config_image_list_filename;
  static sprokit::config::key_t const config_image_list_filename;
  static sprokit::config::value_t const default_image_list_filename;

  // feature tracker algorithm - homography source
  maptk::algo::track_features_sptr m_feature_tracker;
  maptk::algo::compute_ref_homography_sptr m_compute_homog;

  maptk::track_set_sptr m_tracks; // last set of tracks

}; // end priv class

// -- ports --
sprokit::process::port_t const port_timestamp = sprokit::process::port_t("timestamp");
sprokit::process::port_t const port_image = sprokit::process::port_t("image");

sprokit::process::port_t const port_homography = sprokit::process::port_t("src_to_ref_homography");

// ================================================================

stabilize_image_process
::stabilize_image_process( sprokit::config_t const& config )
  : process( config ),
    d( new stabilize_image_process::priv )
{
  make_ports();
  make_config();
}


stabilize_image_process
::~stabilize_image_process()
{
}


// ----------------------------------------------------------------
void stabilize_image_process
::_configure()
{
  // Examine the configuration
  d->m_config_image_list_filename = config_value< std::string > ( stabilize_image_process::priv::config_image_list_filename );

  //+ a better approach is to get the config from this process using get_config()
  // and pass it to the maptk algorithm

  //+ hack - configure file name of maptk config file.

  // Create default maptk config block
  maptk::config_block_sptr config = maptk::config_block::empty_config("stabilize_image");


  // add to config block
  //+ config->set_value( "image_reader:type", d->m_config_image_reader );

  // instantiate image reader and converter based on config type
  maptk::algo::track_features::set_nested_algo_configuration( "feature_tracker", config, d->m_feature_tracker );
  maptk::algo::compute_ref_homography::set_nested_algo_configuration( "homog_generator", config,d->m_compute_homog );

  sprokit::process::_configure();
}


// ----------------------------------------------------------------
void stabilize_image_process
::_step()
{
  maptk::f2f_homography_sptr src_to_ref_homography;

  // timestamp
  kwiver::timestamp frame_time = grab_from_port_as< kwiver::timestamp > ( priv::port_timestamp );

  // image
  maptk::image_container_sptr img = grab_from_port_as< maptk::image_container_sptr > ( priv::port_image );

  // Get feature trac
  d->m_tracks = d->m_feature_tracker->track( d->m_tracks, frame_time.get_frame(), img );

  // Get stabilization homography
  src_to_ref_homography = d->m_compute_homog->estimate( frame_time.get_frame(), d->m_tracks );

  // return by value
  push_to_port_as< maptk::f2f_homography > ( priv::port_homography, *src_to_ref_homography );

  sprokit::process::_step();
}


// ----------------------------------------------------------------
void stabilize_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port(
    priv::port_timestamp,
    kwiver_timestamp,
    required,
    port_description_t( "Timestamp for input image." ) );

  declare_input_port(
    priv::port_image,
    maptk_image_container,
    required,
    port_description_t( "Single frame image." ) );

  // -- output --
  declare_output_port(
    priv::port_homography,
    maptk_src_to_ref_homography,
    required,
    port_description_t( "current to ref image homography." ) );
}


// ----------------------------------------------------------------
void stabilize_image_process
::make_config()
{
  declare_configuration_key(
    priv::config_image_list_filename,
    priv::default_image_list_filename,
    sprokit::config::description_t( "Name of file that contains list of image file names." ));

  // TBD
}


// ================================================================
stabilize_image_process::priv
::priv()
{
}


stabilize_image_process::priv
::~priv()
{
}

} // end namespace
