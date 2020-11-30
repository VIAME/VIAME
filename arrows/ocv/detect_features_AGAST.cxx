// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV AGAST feature detector wrapper
 */

#include "detect_features_AGAST.h"

// Only available in OpenCV 3.x
#if KWIVER_OPENCV_VERSION_MAJOR >= 3

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

namespace {

/**
 * Return multi-line, tabbed list string of available enum types and their values
 */
std::string list_agast_types()
{
  std::stringstream ss;
  ss << "\tAGAST_5_8 = " << cv::AgastFeatureDetector::AGAST_5_8 << "\n"
     << "\tAGAST_7_12d = " << cv::AgastFeatureDetector::AGAST_7_12d << "\n"
     << "\tAGAST_7_12s = " << cv::AgastFeatureDetector::AGAST_7_12s << "\n"
     << "\tOAST_9_16 = " << cv::AgastFeatureDetector::OAST_9_16;
  return ss.str();
}

/**
 * Check that the given integer is one of the valid enum values
 */
bool check_agast_type( int const &type )
{
  switch( type )
  {
    case cv::AgastFeatureDetector::AGAST_5_8:
    case cv::AgastFeatureDetector::AGAST_7_12d:
    case cv::AgastFeatureDetector::AGAST_7_12s:
    case cv::AgastFeatureDetector::OAST_9_16:
      return true;
    default:
      return false;
  }
}

} // end namespace anonymous

class detect_features_AGAST::priv
{
public:
  /// Constructor
  priv()
    : threshold( 10 ),
      nonmax_suppression( true ),
      type( cv::AgastFeatureDetector::OAST_9_16 )
  {
  }

  /// Create algorithm instance
  cv::Ptr<cv::AgastFeatureDetector> create() const
  {
    return cv::AgastFeatureDetector::create( threshold, nonmax_suppression,
                                             type );
  }

  /// Update given algo parameters with currently set values
  void update(cv::Ptr<cv::AgastFeatureDetector> algo) const
  {
    algo->setThreshold( threshold );
    algo->setNonmaxSuppression( nonmax_suppression );
    algo->setType( type );
  }

  /// Update given config block with currently set parameter values
  void update_config( config_block_sptr config ) const
  {
    config->set_value( "threshold", threshold,
                       "Integer threshold on difference between intensity of "
                       "the central pixel and pixels of a circle around this "
                       "pixel" );
    config->set_value( "nonmax_suppression", nonmax_suppression,
                       "if true, non-maximum suppression is applied to "
                       "detected corners (keypoints)" );
    config->set_value( "type", static_cast< int >( type ),
                       "Neighborhood pattern type. Should be one of the "
                       "following enumeration type values:\n"
                       + list_agast_types() + " (default)" );
  }

  /// Set parameter values based on given config block
  void set_config( config_block_sptr const &config )
  {
    threshold = config->get_value<int>( "threshold" );
    nonmax_suppression = config->get_value<bool>( "nonmax_suppression" );
    type = static_cast< decltype( type ) >( config->get_value<int>( "type" ) );
  }

  /// Check config parameter values
  bool check_config(vital::config_block_sptr const &config,
                    logger_handle_t const &logger) const
  {
    bool valid = true;

    int t = config->get_value<int>( "type" );
    if( ! check_agast_type( t ) )
    {
      LOG_ERROR(logger, "Given AGAST type not valid. Must be one of:\n"
                        + list_agast_types() );
      valid = false;
    }

    return valid;
  }

  // Parameters
  int threshold;
  bool nonmax_suppression;
#if KWIVER_OPENCV_VERSION_MAJOR >= 4
  cv::AgastFeatureDetector::DetectorType type;
#else
  int type;
#endif
};

detect_features_AGAST
::detect_features_AGAST()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.AGAST" );
  detector = p_->create();
}

detect_features_AGAST
::~detect_features_AGAST()
{
}

vital::config_block_sptr
detect_features_AGAST
::get_configuration() const
{
  config_block_sptr config = ocv::detect_features::get_configuration();
  p_->update_config( config );
  return config;
}

void
detect_features_AGAST
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
  p_->update( detector.dynamicCast<cv::AgastFeatureDetector>() );
}

bool
detect_features_AGAST
::check_configuration(vital::config_block_sptr config) const
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  return p_->check_config( c, logger() );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //KWIVER_OPENCV_VERSION_MAJOR >= 3
