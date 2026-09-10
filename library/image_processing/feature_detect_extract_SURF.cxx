// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV SURF feature detector and extractor wrapper implementation

#include "feature_detect_extract_SURF.h"

#include <viame/algorithm_framework/vital_config.h>

#ifdef KWIVER_OCV_HAS_SURF

// Include the correct file and unify different namespace locations of SURF type
// across versions
#if KWIVER_OPENCV_VERSION_MAJOR < 3
// 2.4.x header location
#include <opencv2/nonfree/features2d.hpp>
typedef cv::SURF cv_SURF_t;
#else
// 3.x header location
#include <opencv2/xfeatures2d/nonfree.hpp>
typedef cv::xfeatures2d::SURF cv_SURF_t;
#endif

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace ocv {

namespace {

// Create new algorithm instance from current parameters
cv::Ptr< cv_SURF_t >
create(
  double hessian_threshold, int n_octaves, int n_octave_layers,
  bool extended, bool upright )
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  return cv::Ptr< cv_SURF_t >(
    new cv_SURF_t(
      hessian_threshold, n_octaves, n_octave_layers,
      extended, upright  )
  );
#else
  return cv_SURF_t::create(
    hessian_threshold, n_octaves, n_octave_layers,
    extended, upright );
#endif
}

} // end namespace anonymous

// -------------------------------------------------------------------
void
detect_features_SURF
::initialize()
{
  attach_logger( "arrows.ocv.SURF" );
  detector = create(
    get_hessian_threshold(), get_n_octaves(), get_n_octaves_layers(),
    get_extended(), get_upright() );
}

detect_features_SURF
::~detect_features_SURF()
{}

void
detect_features_SURF
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  this->update_detector_parameters();
}

void
detect_features_SURF
::update_detector_parameters() const
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  cv::Ptr< cv_SURF_t > a = this->detector;
  a->set( "hessianThreshold", hessian_threshold );
  a->set( "nOctaves", n_octaves );
  a->set( "nOctaveLayers", n_octave_layers );
  a->set( "extended", extended );
  a->set( "upright", upright );
#else
  // Create a new detector rather than update on version 3.
  // Use of the update function requires a dynamic_cast which fails on Mac
  detector.constCast< cv::FeatureDetector >() = create(
    get_hessian_threshold(), get_n_octaves(), get_n_octaves_layers(),
    get_extended(), get_upright() );
#endif
}

bool
detect_features_SURF
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

// -----------------------------------------------------------------------------
void
extract_descriptors_SURF
::initialize()
{
  attach_logger( "arrows.ocv.SURF" );
  extractor = create(
    get_hessian_threshold(), get_n_octaves(), get_n_octaves_layers(),
    get_extended(), get_upright() );
}

extract_descriptors_SURF
::~extract_descriptors_SURF()
{}

void
extract_descriptors_SURF
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  this->update_extractor_parameters();
}

void
extract_descriptors_SURF
::update_extractor_parameters() const
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  cv::Ptr< cv_SURF_t > a = this->detector;
  a->set( "hessianThreshold", hessian_threshold );
  a->set( "nOctaves", n_octaves );
  a->set( "nOctaveLayers", n_octave_layers );
  a->set( "extended", extended );
  a->set( "upright", upright );
#else
  // Create a new detector rather than update on version 3.
  // Use of the update function requires a dynamic_cast which fails on Mac
  extractor.constCast< cv::DescriptorExtractor >() = create(
    get_hessian_threshold(), get_n_octaves(), get_n_octaves_layers(),
    get_extended(), get_upright() );
#endif
}

bool
extract_descriptors_SURF
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
