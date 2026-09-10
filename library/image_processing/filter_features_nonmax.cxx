// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of filter_features_nonmax algorithm
#include "filter_features_nonmax.h"

#include <viame/core_types/image.h>
#include <viame/core_types/math/aligned_box.h>
#include <viame/algorithm_framework/util/transform_image.h>

#include <algorithm>
#include <cmath>
using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

class nonmax_suppressor
{
public:
  // constructor
  nonmax_suppressor(
    const double suppression_radius,
    aligned_box< double, 2 > feat_bbox,
    const double scale_min,
    const size_t scale_steps,
    const size_t resolution )
    : m_resolution( resolution ),
      m_radius( suppression_radius / resolution ),
      m_feat_bbox( feat_bbox ),
      m_offset( ( -( feat_bbox.min() / m_radius ) ).cwisePlus( 0.5 ) ),
      m_range( ( feat_bbox.sizes() / m_radius ).cwisePlus( 0.5 ) ),
      m_scale_min( scale_min )
  {
    masks.reserve( scale_steps );
    disks.reserve( scale_steps );
    for( size_t s = 0; s < scale_steps; ++s )
    {
      const size_t pad = 2 * resolution + 1;
      const size_t w = ( static_cast< size_t >( m_range[ 0 ] ) >> s ) + pad;
      const size_t h = ( static_cast< size_t >( m_range[ 1 ] ) >> s ) + pad;
      masks.push_back( image_of< bool >( w, h ) );
      // set all pixels in the mask to false
      transform_image( masks.back(), [](bool){ return false; } );

      // create the offsets for each pixel within the circle diameter
      disks.push_back(
        compute_disk_offsets(
          resolution,
          masks.back().w_step(),
          masks.back().h_step() ) );
    }
  }

  // --------------------------------------------------------------------------
  // test if a feature is already covered and if not, cover it
  // return true if the feature was covered by this call
  bool
  cover( feature const& feat )
  {
    // compute the scale and location bin indices
    auto scale =
      static_cast< size_t >( std::log2( feat.scale() ) - m_scale_min );
    vector_2i bin_idx = ( feat.loc() / m_radius + m_offset ).cast< int >();

    // get the center bin at this location from the mask at the current scale
    bool& bin = masks[ scale ]( ( bin_idx[ 0 ] >> scale ) + m_resolution,
                                ( bin_idx[ 1 ] >> scale ) + m_resolution );
    // if the feature is at an uncovered location
    if( !bin )
    {
      // mark all points in a circular neighborhood as covered
      for( auto const& ptr_offset : disks[ scale ] )
      {
        *( &bin + ptr_offset ) = true;
      }
      return true;
    }
    return false;
  }

  // --------------------------------------------------------------------------
  // uncover all pixels in the suppression masks
  void
  uncover_all()
  {
    for( auto& mask : masks )
    {
      transform_image( mask, [](bool){ return false; } );
    }
  }

  void
  set_radius( double r )
  {
    m_radius = r / m_resolution;
    m_offset = ( -( m_feat_bbox.min() / m_radius ) ).cwisePlus( 0.5 );
    m_range = ( m_feat_bbox.sizes() / m_radius ).cwisePlus( 0.5 );

    auto scale_steps = masks.size();
    for( size_t s = 0; s < scale_steps; ++s )
    {
      const size_t pad = 2 * m_resolution + 1;
      const size_t w = ( static_cast< size_t >( m_range[ 0 ] ) >> s ) + pad;
      const size_t h = ( static_cast< size_t >( m_range[ 1 ] ) >> s ) + pad;
      masks[ s ].set_size( w, h, 1 );

      // set all pixels in the mask to false
      transform_image( masks[ s ], [](bool){ return false; } );

      // create the offsets for each pixel within the circle diameter
      disks[ s ] = compute_disk_offsets(
        m_resolution,
        masks[ s ].w_step(),
        masks[ s ].h_step() );
    }
  }

private:
  // --------------------------------------------------------------------------
  std::vector< ptrdiff_t >
  compute_disk_offsets(
    size_t radius, ptrdiff_t w_step, ptrdiff_t h_step ) const
  {
    std::vector< ptrdiff_t > disk;
    const int r = static_cast< int >( radius );
    const int r2 = r * r;
    for( int j = -r; j <= r; ++j )
    {
      const int j2 = j * j;
      for( int i = -r; i <= r; ++i )
      {
        const int i2 = i * i;
        if( ( i2 + j2 ) > r2 )
        {
          continue;
        }
        disk.push_back( j * h_step + i * w_step );
      }
    }
    return disk;
  }

  std::vector< image_of< bool > > masks;
  std::vector< std::vector< ptrdiff_t > > disks;
  size_t m_resolution;
  double m_radius;
  aligned_box< double, 2 > m_feat_bbox;
  vector_2d m_offset;
  vector_2d m_range;
  double m_scale_min;
};

/// Private implementation class
class filter_features_nonmax::priv
{
public:

public:
  priv( filter_features_nonmax& parent )
    : parent( parent )
  {}

  filter_features_nonmax& parent;

  // Configuration Values
  double
  c_suppression_radius() const { return parent.c_suppression_radius; }
  size_t
  c_resolution() const { return parent.c_resolution; }

  size_t
  c_num_features_target() const
  {
    return parent.c_num_features_target;
  }

  size_t
  c_num_features_range() const
  {
    return parent.c_num_features_range;
  }

  // --------------------------------------------------------------------------
  feature_set_sptr
  filter( feature_set_sptr feat_set, std::vector< size_t >& ind ) const
  {
    const std::vector< feature_sptr >& feat_vec = feat_set->features();

    if( feat_vec.size() <= parent.c_num_features_target )
    {
      return feat_set;
    }

    //  Create a new vector with the index and magnitude for faster sorting
    using ud_pair = std::pair< size_t, double >;

    std::vector< ud_pair > indices;
    indices.reserve( feat_vec.size() );

    aligned_box< double, 2 > bbox;
    aligned_box< double, 1 > scale_box;
    for( size_t i = 0; i < feat_vec.size(); i++ )
    {
      auto const& feat = feat_vec[ i ];
      indices.push_back( std::make_pair( i, feat->magnitude() ) );
      bbox.extend( feat->loc() );
      scale_box.extend( vector_< 1, double >( feat->scale() ) );
    }

    const double scale_min = std::log2( scale_box.min()[ 0 ] );
    const double scale_range = std::log2( scale_box.max()[ 0 ] ) - scale_min;
    const size_t scale_steps = static_cast< size_t >( scale_range + 1 );

    LOG_DEBUG(parent.logger(), "Using " << scale_steps << " scale steps" );
    if( scale_steps > 20 )
    {
      LOG_ERROR(
        parent.logger(), "Scale range is too large.  Log2 scales from "
          << scale_box.min() << " to " << scale_box.max() );
      return nullptr;
    }

    if( !bbox.sizes().allFinite() )
    {
      LOG_ERROR(parent.logger(), "Not all features are finite" );
      return nullptr;
    }

    // sort on descending feature magnitude
    std::sort(
      indices.begin(), indices.end(),
      [](const ud_pair& l, const ud_pair& r){
        return l.second > r.second;
      } );

    // compute an upper bound on the radius
    const double& w = bbox.sizes()[ 0 ];
    const double& h = bbox.sizes()[ 1 ];
    const double wph = w + h;
    const double m = parent.c_num_features_target - 1;
    double high_radius = ( wph + std::sqrt( wph * wph + 4 * m * w * h ) ) /
                         ( 2 * m );
    double low_radius = 0.0;

    // initial guess for radius, if not specified
    if( parent.c_suppression_radius <= 0.0 )
    {
      parent.c_suppression_radius = high_radius / 2.0;
    }

    nonmax_suppressor suppressor( parent.c_suppression_radius,
      bbox, scale_min, scale_steps,
      parent.c_resolution );

    // binary search of radius to find the target number of features
    std::vector< feature_sptr > filtered;
    while( true )
    {
      ind.clear();
      filtered.clear();
      filtered.reserve( indices.size() );
      ind.reserve( indices.size() );
      // check each feature against the masks to see if that location
      // has already been covered
      for( auto const& p : indices )
      {
        size_t index = p.first;
        auto const& f = feat_vec[ index ];
        if( suppressor.cover( *f ) )
        {
          // add this feature to the accepted list
          ind.push_back( index );
          filtered.push_back( f );
        }
      }
      // if not using a target number of features, keep this result
      if( parent.c_num_features_target == 0 )
      {
        break;
      }

      // adjust the bounds to continue binary search
      if( filtered.size() < parent.c_num_features_target )
      {
        high_radius = parent.c_suppression_radius;
      }
      else if( filtered.size() >
               parent.c_num_features_target + parent.c_num_features_range )
      {
        low_radius = parent.c_suppression_radius;
      }
      else
      {
        // in the valid range, so we are done
        break;
      }

      double new_suppression_radius = ( high_radius + low_radius ) / 2;
      if( new_suppression_radius < 0.25 )
      {
        LOG_DEBUG(
          parent.logger(), "Found " << filtered.size() << " features.  "
                                                          "Suppression radius is too small to continue." );
        break;
      }
      parent.c_suppression_radius = new_suppression_radius;
      suppressor.set_radius( parent.c_suppression_radius );
      LOG_DEBUG(
        parent.logger(), "Found " << filtered.size() << " features.  "
                                                        "Changing suppression radius to "
                                  << parent.c_suppression_radius);
    }

    LOG_INFO(
      parent.logger(), "Reduced " << feat_vec.size() << " features to "
                                  << filtered.size() <<
        " features with non-max radius "
                                  << parent.c_suppression_radius);

    return std::make_shared< vital::simple_feature_set >(
      vital::simple_feature_set( filtered ) );
  }

private:
};

// ----------------------------------------------------------------------------
void
filter_features_nonmax
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.filter_features_nonmax" );
}

// Destructor
filter_features_nonmax
::~filter_features_nonmax()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
filter_features_nonmax
::check_configuration( vital::config_block_sptr config ) const
{
  size_t resolution =
    config->get_value< size_t >( "resolution", d_->c_resolution() );
  if( resolution < 1 )
  {
    LOG_ERROR(logger(), "resolution must be at least 1" );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
// Filter feature set
vital::feature_set_sptr
filter_features_nonmax
::filter(
  vital::feature_set_sptr feat,
  std::vector< size_t >& indices ) const
{
  return d_->filter( feat, indices );
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
