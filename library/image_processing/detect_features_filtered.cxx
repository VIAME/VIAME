// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of detect_feature_filtered algorithm
#include "detect_features_filtered.h"
#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/filter_features.h>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

/// Private implementation class
class detect_features_filtered::priv
{
public:
  priv( detect_features_filtered& parent )
    : parent( parent )
  {}

  detect_features_filtered& parent;

  // Processing classes
  vital::algo::detect_features_sptr feature_detector()
  { return parent.c_detector; }
  vital::algo::filter_features_sptr feature_filter()
  { return parent.c_filter; }
};

// ----------------------------------------------------------------------------
// Constructor
void
detect_features_filtered
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.detect_features_filtered" );

  d_->parent.logger() = logger();
}

// Destructor
detect_features_filtered
::~detect_features_filtered()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
detect_features_filtered
::check_configuration( vital::config_block_sptr config ) const
{
  bool detector_valid = check_nested_algo_configuration< vital::algo::detect_features >(
    "detector", config );
  bool filter_valid = check_nested_algo_configuration< vital::algo::filter_features >(
    "filter", config );
  return detector_valid && filter_valid;
}

/// Extract a set of image features from the provided image
vital::feature_set_sptr
detect_features_filtered
::detect(
  vital::image_container_sptr image_data,
  vital::image_container_sptr mask ) const
{
  if( !d_->feature_detector() )
  {
    LOG_ERROR(logger(), "Nested feature detector not initialized." );
    return nullptr;
  }

  auto features = d_->feature_detector()->detect( image_data, mask );

  if( !d_->feature_filter() )
  {
    LOG_WARN(logger(), "Nested feature filter not initialized." );
  }
  else
  {
    return d_->feature_filter()->filter( features );
  }
  return features;
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
