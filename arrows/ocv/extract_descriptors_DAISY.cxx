// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV DAISY descriptor extractor wrapper implementation
 */

#include "extract_descriptors_DAISY.h"

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/xfeatures2d.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

namespace {

std::string list_norm_options()
{
  std::stringstream ss;
  ss << "\tNRM_NONE    = " << cv::xfeatures2d::DAISY::NRM_NONE << "\n"
     << "\tNRM_PARTIAL = " << cv::xfeatures2d::DAISY::NRM_PARTIAL << "\n"
     << "\tNRM_FULL    = " << cv::xfeatures2d::DAISY::NRM_FULL << "\n"
     << "\tNRM_SIFT    = " << cv::xfeatures2d::DAISY::NRM_SIFT;
  return ss.str();
}

bool check_norm_type( int norm )
{
  switch( norm )
  {
    case cv::xfeatures2d::DAISY::NRM_NONE:
    case cv::xfeatures2d::DAISY::NRM_PARTIAL:
    case cv::xfeatures2d::DAISY::NRM_FULL:
    case cv::xfeatures2d::DAISY::NRM_SIFT:
      return true;
    default:
      return false;
  }
}

} //end namespace anonymous

class extract_descriptors_DAISY::priv
{
public:
  priv()
    : radius( 15 )
    , q_radius( 3 )
    , q_theta( 3 )
    , q_hist( 8 )
    , norm( cv::xfeatures2d::DAISY::NRM_NONE )
    , interpolation( true )
    , use_orientation( false )
  {
  }

  cv::Ptr<cv::xfeatures2d::DAISY> create() const
  {
    // TODO: Allow custom homography matrix?
    return cv::xfeatures2d::DAISY::create( radius, q_radius, q_theta, q_hist,
                                           norm, cv::noArray(), interpolation,
                                           use_orientation );
  }

  void update_config( config_block_sptr config ) const
  {
    config->set_value( "radius", radius,
                       "radius of the descriptor at the initial scale" );
    config->set_value( "q_radius", q_radius,
                       "amount of radial range division quantity" );
    config->set_value( "q_theta", q_theta,
                       "amount of angular range division quantity" );
    config->set_value( "q_hist", q_hist,
                       "amount of gradient orientations range division quantity" );
    config->set_value( "norm", static_cast< int >( norm ),
                       "descriptor normalization type. valid choices:\n"
                       + list_norm_options() );
    config->set_value( "interpolation", interpolation,
                       "" );
    config->set_value( "use_orientation", use_orientation,
                       "" );
  }

  void set_config( config_block_sptr config )
  {
    radius = config->get_value<float>( "radius" );
    q_radius = config->get_value<int>( "q_radius" );
    q_theta = config->get_value<int>( "q_theta" );
    q_hist = config->get_value<int>( "q_hist" );
    norm = static_cast< decltype( norm ) >( config->get_value<int>( "norm" ) );
    interpolation = config->get_value<bool>( "interpolation" );
    use_orientation = config->get_value<bool>( "use_orientation" );
  }

  bool check_config( config_block_sptr config, logger_handle_t const &log ) const
  {
    bool valid = true;

    int n = config->get_value<int>( "norm" );
    if( ! check_norm_type( n ) )
    {
      LOG_ERROR( log, "Invalid norm option '" << n << "'. Valid choices "
                      "are: " << list_norm_options() );
      valid = false;
    }

    return valid;
  }

  // Parameters
  float radius;
  int q_radius;
  int q_theta;
  int q_hist;
#if KWIVER_OPENCV_VERSION_MAJOR >= 4
  cv::xfeatures2d::DAISY::NormalizationType norm;
#else
  int norm;
#endif
  bool interpolation;
  bool use_orientation;
};

extract_descriptors_DAISY
::extract_descriptors_DAISY()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.DAISY" );
  extractor = p_->create();
}

extract_descriptors_DAISY
::~extract_descriptors_DAISY()
{
}

vital::config_block_sptr
extract_descriptors_DAISY
::get_configuration() const
{
  config_block_sptr config = ocv::extract_descriptors::get_configuration();
  p_->update_config( config );
  return config;
}

void extract_descriptors_DAISY
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
  extractor = p_->create();
}

bool
extract_descriptors_DAISY
::check_configuration(vital::config_block_sptr config) const
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  return p_->check_config( c, logger() );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //HAVE_OPENCV_XFEATURES2D
