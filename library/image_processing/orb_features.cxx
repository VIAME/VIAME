/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_ORB` over the ported algorithm

#include "orb_features.h"

#include "orb.h"

#include <image_kernels/color.h>

#include <viame/core_types/descriptor.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature.h>
#include <viame/core_types/image_container.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace viame {

namespace io = viame::image_kernels;

namespace {

/// The image as one plane of bytes, which is what the algorithm reads.
viame::image_of< uint8_t >
as_grayscale( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    throw std::invalid_argument( "ocv_ORB: no image" );
  }

  viame::image_of< uint8_t > const image( image_data->get_image() );

  if( image.depth() == 1 )
  {
    return image;
  }

  if( image.depth() == 3 )
  {
    return io::rgb_to_gray( image );
  }

  throw std::invalid_argument(
    "ocv_ORB: an image of " + std::to_string( image.depth() ) +
    " planes is neither grayscale nor RGB" );
}

kv::feature_set_sptr
to_feature_set( std::vector< viame::orb::keypoint > const& keypoints )
{
  std::vector< kv::feature_sptr > features;
  std::vector< int > levels;

  features.reserve( keypoints.size() );
  levels.reserve( keypoints.size() );

  for( auto const& kp : keypoints )
  {
    auto feature = std::make_shared< kv::feature_f >();
    feature->set_loc( kv::vector_2f( kp.x, kp.y ) );
    feature->set_magnitude( kp.response );
    feature->set_scale( kp.size );
    feature->set_angle( kp.angle );
    features.push_back( feature );
    levels.push_back( kp.octave );
  }

  return std::make_shared< orb_feature_set >( features, levels );
}

std::vector< viame::orb::keypoint >
from_feature_set( kv::feature_set_sptr features )
{
  std::vector< viame::orb::keypoint > keypoints;

  if( !features )
  {
    return keypoints;
  }

  // Ours, so the levels come back with it; anything else gets zero, which
  // describes every keypoint at the base of the pyramid. See the class
  // comment in orb_features.h.
  auto const* ours = dynamic_cast< orb_feature_set const* >( features.get() );
  auto const all = features->features();

  for( size_t i = 0; i < all.size(); ++i )
  {
    if( !all[ i ] ) { continue; }

    viame::orb::keypoint kp;
    kp.x = static_cast< float >( all[ i ]->loc()[ 0 ] );
    kp.y = static_cast< float >( all[ i ]->loc()[ 1 ] );
    kp.size = static_cast< float >( all[ i ]->scale() );
    kp.angle = static_cast< float >( all[ i ]->angle() );
    kp.response = static_cast< float >( all[ i ]->magnitude() );
    kp.octave = ours && i < ours->levels().size() ? ours->levels()[ i ] : 0;
    keypoints.push_back( kp );
  }

  return keypoints;
}

/// ORB's descriptors are **bytes**, not a float histogram: 32 of them holding
/// 256 single-bit comparisons. They are carried as bytes rather than widened,
/// so that a matcher asked for `binary_descriptors` gets the bit string the
/// Hamming distance is defined over.
kv::descriptor_set_sptr
to_descriptor_set( std::vector< uint8_t > const& raw, int width )
{
  std::vector< kv::descriptor_sptr > descriptors;

  if( width > 0 )
  {
    size_t const count = raw.size() / static_cast< size_t >( width );
    descriptors.reserve( count );

    for( size_t i = 0; i < count; ++i )
    {
      auto descriptor =
        std::make_shared< kv::descriptor_dynamic< uint8_t > >(
          static_cast< size_t >( width ) );
      std::copy( raw.begin() + i * width, raw.begin() + ( i + 1 ) * width,
                 descriptor->raw_data() );
      descriptors.push_back( descriptor );
    }
  }

  return std::make_shared< kv::simple_descriptor_set >( descriptors );
}

viame::orb::settings
as_settings( int features, double scale, int levels, int edge, int first,
             int wta_k, int score, int patch, int fast )
{
  viame::orb::settings settings;
  settings.n_features = features;
  // Narrowed, because `cv::ORB::create` narrows it and the pyramid follows
  // from the narrowed value rather than the double.
  settings.scale_factor = static_cast< float >( scale );
  settings.n_levels = levels;
  settings.edge_threshold = edge;
  settings.first_level = first;
  settings.wta_k = wta_k;
  settings.patch_size = patch;
  settings.fast_threshold = fast;

  // `cv::ORB::HARRIS_SCORE` is 0 and `FAST_SCORE` is 1, and the config key
  // carries the integer rather than a name because that is what the arrow
  // this replaces registered.
  if( score != 0 && score != 1 )
  {
    throw std::invalid_argument(
      "ocv_ORB: score_type must be 0 for HARRIS_SCORE or 1 for FAST_SCORE; "
      "got " + std::to_string( score ) );
  }
  settings.harris_score = score == 0;

  return settings;
}

/// What both algorithms refuse, and why saying so at configure time is
/// better than throwing from `detect` on the first frame.
bool
usable( viame::orb::settings const& settings )
{
  return settings.wta_k == 2 && settings.patch_size == 31 &&
         settings.n_levels >= 1 && settings.first_level >= 0;
}

} // namespace

// ----------------------------------------------------------------------------

void
detect_features_ORB
::initialize()
{
}

detect_features_ORB
::~detect_features_ORB() = default;

bool
detect_features_ORB
::check_configuration( kv::config_block_sptr ) const
{
  try
  {
    return usable( as_settings( c_n_features, c_scale_factor, c_n_levels,
                                c_edge_threshold, c_first_level, c_wta_k,
                                c_score_type, c_patch_size,
                                c_fast_threshold ) );
  }
  catch( std::invalid_argument const& )
  {
    return false;
  }
}

kv::feature_set_sptr
detect_features_ORB
::detect( kv::image_container_sptr image_data,
          kv::image_container_sptr ) const
{
  // The mask is ignored, as the OpenCV arrow's was.
  auto const image = as_grayscale( image_data );

  auto const settings = as_settings( c_n_features, c_scale_factor, c_n_levels,
                                     c_edge_threshold, c_first_level, c_wta_k,
                                     c_score_type, c_patch_size,
                                     c_fast_threshold );

  std::vector< viame::orb::keypoint > keypoints;
  viame::orb::detect_and_compute( image, settings, keypoints, nullptr );

  return to_feature_set( keypoints );
}

// ----------------------------------------------------------------------------

void
extract_descriptors_ORB
::initialize()
{
}

extract_descriptors_ORB
::~extract_descriptors_ORB() = default;

bool
extract_descriptors_ORB
::check_configuration( kv::config_block_sptr ) const
{
  try
  {
    return usable( as_settings( c_n_features, c_scale_factor, c_n_levels,
                                c_edge_threshold, c_first_level, c_wta_k,
                                c_score_type, c_patch_size,
                                c_fast_threshold ) );
  }
  catch( std::invalid_argument const& )
  {
    return false;
  }
}

kv::descriptor_set_sptr
extract_descriptors_ORB
::extract( kv::image_container_sptr image_data,
           kv::feature_set_sptr& features,
           kv::image_container_sptr ) const
{
  if( !image_data || !features )
  {
    return nullptr;
  }

  auto const image = as_grayscale( image_data );

  auto const settings = as_settings( c_n_features, c_scale_factor, c_n_levels,
                                     c_edge_threshold, c_first_level, c_wta_k,
                                     c_score_type, c_patch_size,
                                     c_fast_threshold );

  auto keypoints = from_feature_set( features );
  std::vector< uint8_t > descriptors;

  viame::orb::detect_and_compute( image, settings, keypoints, &descriptors,
                                  true );

  // Replaced rather than left alone, as `ocv_SIFT` replaces it: a keypoint
  // too near a border to describe is dropped from both together, and a caller
  // holding the old array would pair every descriptor after the first drop
  // with the wrong keypoint.
  features = to_feature_set( keypoints );

  return to_descriptor_set( descriptors,
                            viame::orb::descriptor_size( settings ) );
}

} // end namespace viame
