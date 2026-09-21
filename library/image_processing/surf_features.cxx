/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_SURF` over the ported algorithm

#include "surf_features.h"

#include "surf.h"

#include <image_kernels/color.h>

#include <viame/core_types/descriptor.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/image_container.h>

#include <stdexcept>

namespace viame {

namespace io = viame::image_kernels;

namespace {

/// The image as one plane of bytes, which is what the algorithm reads.
///
/// `cv::imread`'s grayscale conversion is `rgb_to_gray`'s, and a caller
/// handing in one plane already is passed through.
viame::image_of< uint8_t >
as_grayscale( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    throw std::invalid_argument( "ocv_SURF: no image" );
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
    "ocv_SURF: an image of " + std::to_string( image.depth() ) +
    " planes is neither grayscale nor RGB" );
}

/// The four fields the bridge has always copied between the two: location,
/// the response as the magnitude, the size as the scale, and the angle.
kv::feature_set_sptr
to_feature_set( std::vector< viame::surf::keypoint > const& keypoints )
{
  std::vector< kv::feature_sptr > features;
  features.reserve( keypoints.size() );

  for( auto const& kp : keypoints )
  {
    auto feature = std::make_shared< kv::feature_f >();
    feature->set_loc( kv::vector_2f( kp.x, kp.y ) );
    feature->set_magnitude( kp.response );
    feature->set_scale( kp.size );
    feature->set_angle( kp.angle );
    features.push_back( feature );
  }

  return std::make_shared< kv::simple_feature_set >( features );
}

std::vector< viame::surf::keypoint >
from_feature_set( kv::feature_set_sptr features )
{
  std::vector< viame::surf::keypoint > keypoints;

  if( !features )
  {
    return keypoints;
  }

  for( auto const& feature : features->features() )
  {
    if( !feature ) { continue; }

    viame::surf::keypoint kp;
    kp.x = static_cast< float >( feature->loc()[ 0 ] );
    kp.y = static_cast< float >( feature->loc()[ 1 ] );
    kp.size = static_cast< float >( feature->scale() );
    kp.angle = static_cast< float >( feature->angle() );
    kp.response = static_cast< float >( feature->magnitude() );
    keypoints.push_back( kp );
  }

  return keypoints;
}

kv::descriptor_set_sptr
to_descriptor_set( std::vector< float > const& raw, int width )
{
  std::vector< kv::descriptor_sptr > descriptors;

  if( width > 0 )
  {
    size_t const count = raw.size() / static_cast< size_t >( width );
    descriptors.reserve( count );

    for( size_t i = 0; i < count; ++i )
    {
      auto descriptor =
        std::make_shared< kv::descriptor_dynamic< float > >(
          static_cast< size_t >( width ) );
      std::copy( raw.begin() + i * width, raw.begin() + ( i + 1 ) * width,
                 descriptor->raw_data() );
      descriptors.push_back( descriptor );
    }
  }

  return std::make_shared< kv::simple_descriptor_set >( descriptors );
}

} // namespace

// ----------------------------------------------------------------------------

void
detect_features_SURF
::initialize()
{
}

detect_features_SURF
::~detect_features_SURF() = default;

bool
detect_features_SURF
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

kv::feature_set_sptr
detect_features_SURF
::detect( kv::image_container_sptr image_data,
          kv::image_container_sptr ) const
{
  // The mask is ignored, as the python implementation ignored it and as the
  // C++ bridge before it passed a null one whatever it was given.
  auto const image = as_grayscale( image_data );

  viame::surf::settings settings;
  settings.hessian_threshold = c_hessian_threshold;
  settings.n_octaves = c_n_octaves;
  settings.n_octaves_layers = c_n_octaves_layers;
  settings.extended = c_extended;
  settings.upright = c_upright;

  std::vector< viame::surf::keypoint > keypoints;
  viame::surf::detect_and_compute( image, settings, keypoints, nullptr );

  return to_feature_set( keypoints );
}

// ----------------------------------------------------------------------------

void
extract_descriptors_SURF
::initialize()
{
}

extract_descriptors_SURF
::~extract_descriptors_SURF() = default;

bool
extract_descriptors_SURF
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

kv::descriptor_set_sptr
extract_descriptors_SURF
::extract( kv::image_container_sptr image_data,
           kv::feature_set_sptr& features,
           kv::image_container_sptr ) const
{
  if( !image_data || !features )
  {
    return nullptr;
  }

  auto const image = as_grayscale( image_data );

  viame::surf::settings settings;
  settings.hessian_threshold = c_hessian_threshold;
  settings.n_octaves = c_n_octaves;
  settings.n_octaves_layers = c_n_octaves_layers;
  settings.extended = c_extended;
  settings.upright = c_upright;

  auto keypoints = from_feature_set( features );
  std::vector< float > descriptors;

  // Describing given keypoints, which also rewrites their orientation and
  // drops any too near a border. `features` is replaced rather than left
  // alone: a caller holding the old set would pair every descriptor after
  // the first drop with the wrong feature.
  viame::surf::detect_and_compute( image, settings, keypoints, &descriptors,
                                   true );

  features = to_feature_set( keypoints );

  return to_descriptor_set( descriptors,
                            viame::surf::descriptor_size( settings ) );
}

} // end namespace viame
