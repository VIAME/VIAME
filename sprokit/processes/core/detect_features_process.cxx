// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detect_features_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/detect_features.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( feature_detector );

/**
 * \class detect_features_process
 *
 * \brief Detect feature points in supplied images.
 *
 * \process This process generates a list of detected features that
 * can be used to determine coordinate transforms between images. The
 * actual rendering is done by the selected \b detect_features
 * algorithm implementation
 *
 * \iports
 *
 * \iport{timestamp} time stamp for incoming images.
 *
 * \iport{image} Input image to be processed.
 *
 * \oports
 *
 * \oport{feature_set} Set of detected features for input image.
 *
 * \configs
 *
 * \config{feature_detector} Name of the configuration subblock that selects
 * and configures the feature detector algorithm
 */

//----------------------------------------------------------------
// Private implementation class
class detect_features_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  algo::detect_features_sptr m_detector;

}; // end priv class

// ================================================================

detect_features_process
::detect_features_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detect_features_process::priv )
{
  make_ports();
  make_config();
}

detect_features_process
::~detect_features_process()
{
}

// ----------------------------------------------------------------
void detect_features_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::detect_features::check_nested_algo_configuration_using_trait(
         feature_detector, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // Instantiate the configured algorithm
  algo::detect_features::set_nested_algo_configuration_using_trait(
    feature_detector,
    algo_config,
    d->m_detector );
  if ( ! d->m_detector )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create feature_detector" );
  }
}

// ----------------------------------------------------------------
void
detect_features_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  kwiver::vital::feature_set_sptr curr_feat;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    // detect features on the current frame
    curr_feat = d->m_detector->detect( img );
  }

  // return by value
  push_to_port_using_trait( feature_set, curr_feat );
}

// ----------------------------------------------------------------
void detect_features_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( feature_set, optional );
}

// ----------------------------------------------------------------
void detect_features_process
::make_config()
{
  declare_config_using_trait( feature_detector );
}

// ================================================================
detect_features_process::priv
::priv()
{
}

detect_features_process::priv
::~priv()
{
}

} // end namespace
