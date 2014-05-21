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
#include <core/config_util.h>

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

  // There are many config items for the tracking and stabilization that go directly to
  // the maptk algo.

  // feature tracker algorithm - homography source
  maptk::algo::track_features_sptr m_feature_tracker;
  maptk::algo::compute_ref_homography_sptr m_compute_homog;

  maptk::track_set_sptr m_tracks; // last set of tracks

}; // end priv class

// -- config --

// -- ports --
sprokit::process::port_t const stabilize_image_process::priv::port_timestamp = sprokit::process::port_t("timestamp");
sprokit::process::port_t const stabilize_image_process::priv::port_image = sprokit::process::port_t("image");

sprokit::process::port_t const stabilize_image_process::priv::port_homography = sprokit::process::port_t("src_to_ref_homography");

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
  // Convert sprokit config to maptk config for algorithms
  sprokit::config_t proc_config = get_config(); // config for process
  maptk::config_block_sptr algo_config = maptk::config_block::empty_config();

  convert_config( proc_config, algo_config );

  // instantiate image reader and converter based on config type
  maptk::algo::track_features::set_nested_algo_configuration( "feature_tracker",
                                                algo_config, d->m_feature_tracker );
  maptk::algo::compute_ref_homography::set_nested_algo_configuration( "homography_generator",
                                                          algo_config, d->m_compute_homog );

  sprokit::process::_configure();
}


// ----------------------------------------------------------------
void stabilize_image_process
::_step()
{
  maptk::f2f_homography_sptr src_to_ref_homography;

  // timestamp
  kwiver::timestamp frame_time = grab_input_as< kwiver::timestamp > ( priv::port_timestamp );

  // image
  maptk::image_container_sptr img = grab_input_as< maptk::image_container_sptr > ( priv::port_image );

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
