// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV Star feature detector wrapper implementation
 */

#include "detect_features_STAR.h"

#if KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)

#include <vital/vital_config.h>

#if KWIVER_OPENCV_VERSION_MAJOR < 3
typedef cv::StarDetector cv_STAR_t;
#else
#include <opencv2/xfeatures2d.hpp>
typedef cv::xfeatures2d::StarDetector cv_STAR_t;
#endif

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

class detect_features_STAR::priv
{
public:
  priv()
    : max_size( 45 )
    , response_threshold( 30 )
    , line_threshold_projected( 10 )
    , line_threshold_binarized( 8 )
    , suppress_nonmax_size( 5 )
  {
  }

  cv::Ptr<cv_STAR_t> create() const
  {
#if KWIVER_OPENCV_VERSION_MAJOR < 3
    return cv::Ptr<cv_STAR_t>(
      new cv_STAR_t( max_size, response_threshold, line_threshold_projected,
                     line_threshold_binarized, suppress_nonmax_size )
    );
#else
    return cv_STAR_t::create( max_size, response_threshold,
                              line_threshold_projected,
                              line_threshold_binarized, suppress_nonmax_size );
#endif
  }

#if KWIVER_OPENCV_VERSION_MAJOR < 3
  void update( cv::Ptr<cv_STAR_t> a ) const
  {
    a->set( "maxSize", max_size );
    a->set( "responseThreshold", response_threshold );
    a->set( "lineThresholdProjected", line_threshold_projected );
    a->set( "lineThresholdBinarized", line_threshold_binarized );
    a->set( "suppressNonmaxSize", suppress_nonmax_size );
  }
#endif

  void update_config( config_block_sptr config ) const
  {
    config->set_value( "max_size", max_size );
    config->set_value( "response_threshold", response_threshold );
    config->set_value( "line_threshold_projected", line_threshold_projected );
    config->set_value( "line_threshold_binarized", line_threshold_binarized );
    config->set_value( "suppress_nonmax_size", suppress_nonmax_size );
  }

  void set_config( config_block_sptr config )
  {
    max_size = config->get_value<int>( "max_size" );
    response_threshold = config->get_value<int>( "response_threshold" );
    line_threshold_projected = config->get_value<int>( "line_threshold_projected" );
    line_threshold_binarized = config->get_value<int>( "line_threshold_binarized" );
    suppress_nonmax_size = config->get_value<int>( "suppress_nonmax_size" );
  }

  // Parameters
  int max_size;
  int response_threshold;
  int line_threshold_projected;
  int line_threshold_binarized;
  int suppress_nonmax_size;
};

detect_features_STAR
::detect_features_STAR()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.star" );
  detector = p_->create();
}

detect_features_STAR
::~detect_features_STAR()
{
}

vital::config_block_sptr
detect_features_STAR
::get_configuration() const
{
  config_block_sptr config = ocv::detect_features::get_configuration();
  p_->update_config( config );
  return config;
}

void
detect_features_STAR
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  p_->update( detector );
#else
  detector = p_->create();
#endif
}

bool
detect_features_STAR
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //has OCV support
