// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV GFTT feature detector wrapper implementation
 */

#include "detect_features_GFTT.h"

#include <vital/vital_config.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

class detect_features_GFTT::priv
{
public:
  /// Constructor
  priv()
    : max_corners( 1000 ),
      quality_level( 0.01 ),
      min_distance( 1.0 ),
      block_size( 3 ),
      use_harris_detector( false ),
      k( 0.04 )
  {
  }

  /// Create a new GFTT detector instance with the current parameter values
  cv::Ptr<cv::GFTTDetector> create() const
  {
#if KWIVER_OPENCV_VERSION_MAJOR < 3
    return cv::Ptr<cv::GFTTDetector>(
      new cv::GFTTDetector( max_corners, quality_level, min_distance,
                            block_size, use_harris_detector, k )
    );
#else
    return cv::GFTTDetector::create( max_corners, quality_level, min_distance,
                                     block_size, use_harris_detector, k );
#endif
  }

#if KWIVER_OPENCV_VERSION_MAJOR >= 3
  /// Update the parameters of the given detector with the currently set values
  /**
   * Returns false if the algo could not be updating, requiring recreation.
   */
  bool update(cv::Ptr<cv::GFTTDetector> a) const
  {
    a->setMaxFeatures( max_corners );
    a->setQualityLevel( quality_level );
    a->setMinDistance( min_distance );
    a->setBlockSize( block_size );
    a->setHarrisDetector( use_harris_detector );
    a->setK( k );
    return true;
  }
#endif

  /// Update given config block with currently set parameter values
  void update_config( config_block_sptr config ) const
  {
    config->set_value( "max_corners", max_corners );
    config->set_value( "quality_level", quality_level );
    config->set_value( "min_distance", min_distance );
    config->set_value( "block_size", block_size );
    config->set_value( "use_harris_detector", use_harris_detector );
    config->set_value( "k", k );
  }

  /// Set parameter values based on given config block
  void set_config( config_block_sptr const &config )
  {
    max_corners = config->get_value<int>( "max_corners" );
    quality_level = config->get_value<double>( "quality_level" );
    min_distance = config->get_value<double>( "min_distance" );
    block_size = config->get_value<int>( "block_size" );
    use_harris_detector = config->get_value<bool>( "use_harris_detector" );
    k = config->get_value<double>( "k" );
  }

  /// Parameters
  int max_corners;
  double quality_level;
  double min_distance;
  int block_size;
  bool use_harris_detector;
  double k;
};

detect_features_GFTT
::detect_features_GFTT()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.GFTT" );
  detector = p_->create();
}

detect_features_GFTT
::~detect_features_GFTT()
{
}

vital::config_block_sptr
detect_features_GFTT
::get_configuration() const
{
  config_block_sptr config = ocv::detect_features::get_configuration();
  p_->update_config( config );
  return config;
}

void
detect_features_GFTT
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  // since 2.4.x does not have params set for everything that's given to the
  // constructor, lets just remake the algo instance.
  detector = p_->create();
#else
  p_->update( detector.dynamicCast<cv::GFTTDetector>() );
#endif
}

bool
detect_features_GFTT
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  // Nothing to explicitly check
  return true;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
