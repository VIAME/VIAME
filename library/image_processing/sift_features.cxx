/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_SIFT` over the ported algorithm

#include "sift_features.h"

#include "sift.h"

#include <image_kernels/color.h>

#include <viame/core_types/descriptor.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature.h>
#include <viame/core_types/image_container.h>

#include <stdexcept>

namespace viame {

namespace io = viame::image_kernels;

namespace {

/// The image as one plane of bytes, which is what the algorithm reads.
viame::image_of< uint8_t >
as_grayscale( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    throw std::invalid_argument( "ocv_SIFT: no image" );
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
    "ocv_SIFT: an image of " + std::to_string( image.depth() ) +
    " planes is neither grayscale nor RGB" );
}

kv::feature_set_sptr
to_feature_set( std::vector< viame::sift::keypoint > const& keypoints )
{
  std::vector< kv::feature_sptr > features;
  std::vector< int > octaves;

  features.reserve( keypoints.size() );
  octaves.reserve( keypoints.size() );

  for( auto const& kp : keypoints )
  {
    auto feature = std::make_shared< kv::feature_f >();
    feature->set_loc( kv::vector_2f( kp.x, kp.y ) );
    feature->set_magnitude( kp.response );
    feature->set_scale( kp.size );
    feature->set_angle( kp.angle );
    features.push_back( feature );
    octaves.push_back( kp.octave );
  }

  return std::make_shared< sift_feature_set >( features, octaves );
}

std::vector< viame::sift::keypoint >
from_feature_set( kv::feature_set_sptr features )
{
  std::vector< viame::sift::keypoint > keypoints;

  if( !features )
  {
    return keypoints;
  }

  // Ours, so the octaves come back with it; anything else gets zero, which
  // describes every keypoint at the base of the pyramid. See the class comment
  // in sift_features.h.
  auto const* ours = dynamic_cast< sift_feature_set const* >( features.get() );
  auto const all = features->features();

  for( size_t i = 0; i < all.size(); ++i )
  {
    if( !all[ i ] ) { continue; }

    viame::sift::keypoint kp;
    kp.x = static_cast< float >( all[ i ]->loc()[ 0 ] );
    kp.y = static_cast< float >( all[ i ]->loc()[ 1 ] );
    kp.size = static_cast< float >( all[ i ]->scale() );
    kp.angle = static_cast< float >( all[ i ]->angle() );
    kp.response = static_cast< float >( all[ i ]->magnitude() );
    kp.octave = ours && i < ours->octaves().size()
                ? ours->octaves()[ i ] : 0;
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

viame::sift::settings
as_settings( int features, int layers, double contrast, int edge,
             double sigma )
{
  viame::sift::settings settings;
  settings.n_features = features;
  settings.n_octave_layers = layers;
  settings.contrast_threshold = contrast;
  settings.edge_threshold = static_cast< double >( edge );
  settings.sigma = sigma;

  return settings;
}

} // namespace

// ----------------------------------------------------------------------------

void
detect_features_SIFT
::initialize()
{
}

detect_features_SIFT
::~detect_features_SIFT() = default;

bool
detect_features_SIFT
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

kv::feature_set_sptr
detect_features_SIFT
::detect( kv::image_container_sptr image_data,
          kv::image_container_sptr ) const
{
  // The mask is ignored, as the python implementation ignored it.
  auto const image = as_grayscale( image_data );

  auto const settings = as_settings( c_n_features, c_n_octave_layers,
                                     c_contrast_threshold, c_edge_threshold,
                                     c_sigma );

  std::vector< viame::sift::keypoint > keypoints;
  viame::sift::detect_and_compute( image, settings, keypoints, nullptr );

  return to_feature_set( keypoints );
}

// ----------------------------------------------------------------------------

void
extract_descriptors_SIFT
::initialize()
{
}

extract_descriptors_SIFT
::~extract_descriptors_SIFT() = default;

bool
extract_descriptors_SIFT
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

kv::descriptor_set_sptr
extract_descriptors_SIFT
::extract( kv::image_container_sptr image_data,
           kv::feature_set_sptr& features,
           kv::image_container_sptr ) const
{
  if( !image_data || !features )
  {
    return nullptr;
  }

  auto const image = as_grayscale( image_data );

  auto const settings = as_settings( c_n_features, c_n_octave_layers,
                                     c_contrast_threshold, c_edge_threshold,
                                     c_sigma );

  auto keypoints = from_feature_set( features );
  std::vector< float > descriptors;

  viame::sift::detect_and_compute( image, settings, keypoints, &descriptors,
                                   true );

  // Replaced rather than left alone, as the python implementation replaced it:
  // the described set is the one each descriptor belongs to.
  features = to_feature_set( keypoints );

  return to_descriptor_set( descriptors, viame::sift::descriptor_size() );
}

} // end namespace viame
