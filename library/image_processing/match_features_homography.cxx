// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of the core match_features_homography algorithm

#include "match_features_homography.h"

#include <iostream>

#include <viame/algorithm_framework/exceptions/algorithm.h>
#include <viame/core_types/homography.h>
#include <viame/core_types/match_set.h>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

// Private implementation class
class match_features_homography::priv
{
public:
  // Constructor
  priv( match_features_homography& parent )
    : parent( parent )
  {}

  match_features_homography& parent;

  // Configuration values
  double c_inlier_scale() { return parent.c_inlier_scale; }
  int c_min_required_inlier_count()
  { return parent.c_min_required_inlier_count; }
  double c_min_required_inlier_percent()
  { return parent.c_min_required_inlier_percent; }

  // processing classes
  vital::algo::estimate_homography_sptr c_h_estimator()
  { return parent.c_homography_estimator; }

  vital::algo::match_features_sptr
  c_matcher1()
  {
    return parent.c_feature_matcher1;
  }

  vital::algo::match_features_sptr
  c_matcher2()
  {
    return parent.c_feature_matcher2;
  }

  vital::algo::filter_features_sptr c_feature_filter()
  { return parent.c_filter_features; }
};

// ----------------------------------------------------------------------------
void
match_features_homography
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.match_features_homography" );
}

// Destructor
match_features_homography
::~match_features_homography()
{}

// ----------------------------------------------------------------------------
bool
match_features_homography
::check_configuration( vital::config_block_sptr config ) const
{
  bool config_valid = true;
  // this algorithm is optional
  if( config->has_value( "filter_features" ) &&
      config->get_value< std::string >( "filter_features" ) != "" &&
      !check_nested_algo_configuration< vital::algo::filter_features >(
        "filter_features", config ) )
  {
    config_valid = false;
  }
  // this algorithm is optional
  if( config->has_value( "feature_matcher2" ) &&
      config->get_value< std::string >( "feature_matcher2" ) != "" &&
      !check_nested_algo_configuration< vital::algo::match_features >(
        "feature_matcher2", config ) )
  {
    config_valid = false;
  }
  return (
    check_nested_algo_configuration< vital::algo::estimate_homography >(
      "homography_estimator", config )
    &&
    check_nested_algo_configuration< vital::algo::match_features >(
      "feature_matcher1", config )
    &&
    config_valid
  );
}

// ----------------------------------------------------------------------------
namespace {

// Compute the average feature scale
double
average_feature_scale( feature_set_sptr features )
{
  double scale = 0.0;
  if( !features )
  {
    return scale;
  }
  for( auto const& f : features->features() )
  {
    scale += f->scale();
  }
  if( features->size() > 0 )
  {
    scale /= features->size();
  }
  return scale;
}

// Compute the minimum feature scale
double
min_feature_scale( feature_set_sptr features )
{
  double min_scale = std::numeric_limits< double >::infinity();
  if( !features || features->size() == 0 )
  {
    return 1.0;
  }
  for( auto const& f : features->features() )
  {
    min_scale = std::min( min_scale, f->scale() );
  }
  return min_scale;
}

} // namespace

// Match one set of features and corresponding descriptors to another
match_set_sptr
match_features_homography
::match(
  feature_set_sptr feat1, descriptor_set_sptr desc1,
  feature_set_sptr feat2, descriptor_set_sptr desc2 ) const
{
  if( !d_->c_matcher1() || !d_->c_h_estimator() )
  {
    return match_set_sptr();
  }

  // filter features if a filter_features is set
  feature_set_sptr src_feat = feat1,  dst_feat = feat2;
  descriptor_set_sptr src_desc = desc1,  dst_desc = desc2;
  if( d_->c_feature_filter().get() )
  {
    // filter source image features
    auto ret = d_->c_feature_filter()->filter( feat1, desc1 );
    src_feat = ret.first;
    src_desc = ret.second;

    // filter destination image features
    ret = d_->c_feature_filter()->filter( feat2, desc2 );
    dst_feat = ret.first;
    dst_desc = ret.second;
  }

  double avg_scale = ( average_feature_scale( src_feat ) +
                       average_feature_scale( dst_feat ) ) / 2.0;

  // ideally the notion of scale would be standardized relative to
  // some baseline, regardless of the detector, but currently it is not
  // so we get the minimum observed scale in the data
  double min_scale = std::min(
    min_feature_scale( feat1 ),
    min_feature_scale( feat2 ) );

  double scale_ratio = avg_scale / min_scale;
  LOG_DEBUG( logger(), "Filtered scale ratio: " << scale_ratio );

  // compute the initial matches
  match_set_sptr init_matches = d_->c_matcher1()->match(
    src_feat, src_desc,
    dst_feat, dst_desc );

  // estimate a homography from the initial matches
  std::vector< bool > inliers;
  homography_sptr H = d_->c_h_estimator()->estimate(
    src_feat, dst_feat, init_matches,
    inliers, d_->c_inlier_scale() * scale_ratio );

  // count the number of inliers
  int inlier_count = static_cast< int >( std::count(
    inliers.begin(),
    inliers.end(), true ) );
  LOG_INFO(
    logger(), "inlier ratio: " <<
      inlier_count << "/" << inliers.size() );

  // verify matching criteria are met
  if( !inlier_count || inlier_count < d_->c_min_required_inlier_count() ||
      static_cast< double >( inlier_count ) / inliers.size() <
      d_->c_min_required_inlier_percent() )
  {
    return match_set_sptr( new simple_match_set() );
  }

  if( !d_->c_matcher2() )
  {
    // return the subset of inlier matches
    std::vector< vital::match > m = init_matches->matches();
    std::vector< vital::match > inlier_m;
    for( size_t i = 0; i < inliers.size(); ++i )
    {
      if( inliers[ i ] )
      {
        inlier_m.push_back( m[ i ] );
      }
    }

    return match_set_sptr( new simple_match_set( inlier_m ) );
  }

  // deep copy and warp the original (non filtered) points
  const std::vector< feature_sptr >& feat1_vec = feat1->features();
  std::vector< feature_sptr > warped_feat1;
  warped_feat1.reserve( feat1_vec.size() );

  homography_< double > Hd( *H );
  for( size_t i = 0; i < feat1_vec.size(); i++ )
  {
    feature_< double > f( *feat1_vec[ i ] );
    f.set_loc( Hd.map_point( f.get_loc() ) );
    warped_feat1.push_back( std::make_shared< feature_< double > >( f ) );
  }

  feature_set_sptr warped_feat1_set =
    std::make_shared< simple_feature_set >(
      simple_feature_set( warped_feat1 ) );

  return d_->c_matcher2()->match( warped_feat1_set, desc1, feat2, desc2 );
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
