// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV FREAK descriptor extractor wrapper implementation
 */

#include "extract_descriptors_FREAK.h"

#if KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)

#include <vital/vital_config.h>

// typedef FREAK into a common symbol
#if KWIVER_OPENCV_VERSION_MAJOR < 3
typedef cv::FREAK cv_FREAK_t;
#else
#include <opencv2/xfeatures2d.hpp>
typedef cv::xfeatures2d::FREAK cv_FREAK_t;
#endif

namespace kwiver {
namespace arrows {
namespace ocv {

class extract_descriptors_FREAK::priv
{
public:
  /// Constructor
  priv()
    : orientation_normalized( true ),
      scale_normalized( true ),
      pattern_scale( 22.0f ),
      n_octaves( 4 )
  {
  }

  /// Create new cv::Ptr algo instance
  cv::Ptr<cv_FREAK_t> create() const
  {
#if KWIVER_OPENCV_VERSION_MAJOR < 3
    return cv::Ptr<cv_FREAK_t>(
        new cv_FREAK_t( orientation_normalized, scale_normalized, pattern_scale,
                        n_octaves )
    );
#else
    return cv_FREAK_t::create( orientation_normalized, scale_normalized,
                               pattern_scale, n_octaves );
#endif
  }

#if KWIVER_OPENCV_VERSION_MAJOR < 3
  /// Update algorithm instance with current parameter values
  void update( cv::Ptr<cv_FREAK_t> freak ) const
  {
    freak->set( "orientationNormalized", orientation_normalized );
    freak->set( "scaleNormalized", scale_normalized );
    freak->set( "patternScale", pattern_scale );
    freak->set( "nbOctave", n_octaves );
  }
#endif

  /// Set current parameter values to the given config block
  void update_config( vital::config_block_sptr &config ) const
  {
    config->set_value( "orientation_normalized", orientation_normalized,
                       "enable orientation normalization" );
    config->set_value( "scale_normalized", scale_normalized,
                       "enable scale normalization" );
    config->set_value( "pattern_scale", pattern_scale,
                       "scaling of the description pattern" );
    config->set_value( "n_octaves", n_octaves,
                       "number of octaves covered by the detected keypoints" );
  }

  /// Set our parameters based on the given config block
  void set_config( vital::config_block_sptr const &config )
  {
    orientation_normalized = config->get_value<bool>("orientation_normalized");
    scale_normalized = config->get_value<bool>("scale_normalized");
    pattern_scale = config->get_value<float>("pattern_scale");
    n_octaves = config->get_value<int>("n_octaves");
  }

  /// Params
  bool orientation_normalized;
  bool scale_normalized;
  float pattern_scale;
  int n_octaves;
};

/// Constructor
extract_descriptors_FREAK
::extract_descriptors_FREAK()
    : p_( new priv )
{
  attach_logger("arrows.ocv.FREAK");
  extractor = p_->create();
}

/// Destructor
extract_descriptors_FREAK
::~extract_descriptors_FREAK()
{
}

vital::config_block_sptr
extract_descriptors_FREAK
::get_configuration() const
{
  vital::config_block_sptr config = extract_descriptors::get_configuration();
  p_->update_config(config);
  return config;
}

void
extract_descriptors_FREAK
::set_configuration(vital::config_block_sptr config)
{
  vital::config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  p_->update( extractor );
#else
  extractor = p_->create();
#endif
}

bool
extract_descriptors_FREAK
::check_configuration( VITAL_UNUSED vital::config_block_sptr in_config ) const
{
  return true;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)
