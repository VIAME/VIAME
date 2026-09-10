// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV SIFT feature detector and extractor wrapper implementation

#include "feature_detect_extract_SIFT.h"

#include <viame/algorithm_framework/vital_config.h>

#if defined( HAVE_OPENCV_NONFREE ) || defined( HAVE_OPENCV_XFEATURES2D )

#include <opencv2/core/version.hpp>

// Include the correct file and unify different namespace locations of SIFT type
// across versions
#if CV_VERSION_MAJOR < 3
// 2.4.x header location
#include <opencv2/nonfree/features2d.hpp>
typedef cv::SIFT cv_SIFT_t;
#elif CV_VERSION_MAJOR < 4 || ( CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 3 )
// 3.x - 4.2 header location
#include <opencv2/xfeatures2d/nonfree.hpp>
typedef cv::xfeatures2d::SIFT cv_SIFT_t;
#else
// 4.4+ header location
#include <opencv2/features2d.hpp>
typedef cv::SIFT cv_SIFT_t;
#endif

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace ocv {

namespace {

cv::Ptr< cv_SIFT_t >
create(
  int n_features, int n_octave_layers, double contrast_threshold,
  int edge_threshold, double sigma )
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  return cv::Ptr< cv_SIFT_t >(
    new cv_SIFT_t(
      n_features, n_octave_layers, contrast_threshold,
      edge_threshold, sigma )
  );
#else
  return cv_SIFT_t::create(
    n_features, n_octave_layers, contrast_threshold,
    edge_threshold, sigma );
#endif
}

} // namespace

// --------------------------------------------------------------------
void
detect_features_SIFT
::initialize()
{
  attach_logger( "arrows.ocv.SIFT" );
  this->detector = create(
    this->get_n_features(),
    this->get_n_octave_layers(),
    this->get_contrast_threshold(),
    this->get_edge_threshold(),
    this->get_sigma() );
}

detect_features_SIFT
::~detect_features_SIFT()
{}

void
detect_features_SIFT
::update_detector_parameters() const
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  cv::Ptr< cv_SIFT_t > a = this->detector;
  a->set( "nFeatures", get_n_features() );
  a->set( "nOctaveLayers", get_n_octave_layers() );
  a->set( "contrastThreshold", get_contrast_threshold() );
  a->set( "edgeThreshold", get_edge_threshold() );
  a->set( "sigma", get_sigma() );
#else
  // version 3.x doesn't have parameter update methods
  detector.constCast< cv::FeatureDetector >() = create(
    this->get_n_features(),
    this->get_n_octave_layers(),
    this->get_contrast_threshold(),
    this->get_edge_threshold(),
    this->get_sigma() );
#endif
}

void
detect_features_SIFT
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  this->update_detector_parameters();
}

bool
detect_features_SIFT
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

// --------------------------------------------------------------------
void
extract_descriptors_SIFT
::initialize()
{
  attach_logger( "arrows.ocv.SIFT" );
  this->extractor = create(
    this->get_n_features(),
    this->get_n_octave_layers(),
    this->get_contrast_threshold(),
    this->get_edge_threshold(),
    this->get_sigma() );
}

extract_descriptors_SIFT
::~extract_descriptors_SIFT()
{}

void
extract_descriptors_SIFT
::update_extractor_parameters() const
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  cv::Ptr< cv_SIFT_t > a = this->detector;
  a->set( "nFeatures", get_n_features() );
  a->set( "nOctaveLayers", get_n_octave_layers() );
  a->set( "contrastThreshold", get_contrast_threshold() );
  a->set( "edgeThreshold", get_edge_threshold() );
  a->set( "sigma", get_sigma() );
#else
  // version 3.x doesn't have parameter update methods
  extractor.constCast< cv::DescriptorExtractor >() = create(
    this->get_n_features(),
    this->get_n_octave_layers(),
    this->get_contrast_threshold(),
    this->get_edge_threshold(),
    this->get_sigma() );
#endif
}

void
extract_descriptors_SIFT
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  this->update_extractor_parameters();
}

bool
extract_descriptors_SIFT
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif // defined(HAVE_OPENCV_NONFREE) || defined(HAVE_OPENCV_XFEATURES2D)
