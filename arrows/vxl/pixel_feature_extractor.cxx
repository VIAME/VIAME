// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pixel_feature_extractor.h"

#include "aligned_edge_detection.h"
#include "average_frames.h"
#include "color_commonality_filter.h"
#include "convert_image.h"
#include "high_pass_filter.h"

#include <arrows/vxl/image_container.h>

#include <vital/config/config_block_io.h>
#include <vital/range/iota.h>

#include <vil/vil_clamp.h>
#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>

#include <cstdlib>
#include <limits>
#include <type_traits>

namespace kwiver {

namespace arrows {

namespace vxl {

namespace kvr = kwiver::vital::range;

// ----------------------------------------------------------------------------
// Private implementation class
class pixel_feature_extractor::priv
{
public:
  priv( pixel_feature_extractor* parent ) : p{ parent }
  {
  }

  // Check the configuration of the sub algoirthms
  bool check_sub_algorithm( vital::config_block_sptr config, std::string key );
  // Generate the spatial encoding image
  vil_image_view< vxl_byte >
  generate_spatial_prior( kwiver::vital::image_container_sptr input_image );
  // Copy multiple filtered images into contigious memory
  template < typename pix_t > vil_image_view< pix_t >
  concatenate_images( std::vector< vil_image_view< pix_t > > filtered_images );
  // Extract local pixel-wise features
  template < typename response_t > vil_image_view< response_t >
  filter( kwiver::vital::image_container_sptr input_image );

  pixel_feature_extractor* p;

  unsigned frame_number{ 0 };
  vil_image_view< vxl_byte > spatial_prior;

  bool enable_color{ true };
  bool enable_gray{ true };
  bool enable_aligned_edge{ true };
  bool enable_average{ true };
  bool enable_color_commonality{ true };
  bool enable_high_pass_bidir{ true };
  bool enable_high_pass_box{ true };
  bool enable_normalized_variance{ true };
  bool enable_spatial_prior{ true };

  float variance_scale_factor{ 0.32f };
  unsigned grid_length{ 5 };

  std::shared_ptr< vxl::aligned_edge_detection >
  aligned_edge_detection_filter =
    std::make_shared< vxl::aligned_edge_detection >();
  std::shared_ptr< vxl::average_frames > average_frames_filter =
    std::make_shared< vxl::average_frames >();
  std::shared_ptr< vxl::color_commonality_filter > color_commonality_filter =
    std::make_shared< vxl::color_commonality_filter >();
  std::shared_ptr< vxl::high_pass_filter > high_pass_bidir_filter =
    std::make_shared< vxl::high_pass_filter >();
  std::shared_ptr< vxl::high_pass_filter > high_pass_box_filter =
    std::make_shared< vxl::high_pass_filter >();

  std::map< std::string,
            std::shared_ptr< vital::algo::image_filter > > filters{
    std::make_pair( "aligned_edge", aligned_edge_detection_filter ),
    std::make_pair( "average", average_frames_filter ),
    std::make_pair( "color_commonality", color_commonality_filter ),
    std::make_pair( "high_pass_bidir", high_pass_bidir_filter ),
    std::make_pair( "high_pass_box", high_pass_box_filter ) };
};

// ----------------------------------------------------------------------------
bool
pixel_feature_extractor::priv
::check_sub_algorithm( vital::config_block_sptr config, std::string key )
{
  auto enabled = config->get_value< bool >( "enable_" + key );

  if( !enabled )
  {
    return true;
  }
  auto subblock = config->subblock_view( key );
  if( !filters.at( key )->check_configuration( subblock ) )
  {
    LOG_ERROR(
      p->logger(),
      "Sub-algorithm " << key << " failed its config check" );
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
vil_image_view< vxl_byte >
pixel_feature_extractor::priv
::generate_spatial_prior( kwiver::vital::image_container_sptr input_image )
{
  auto const ni = static_cast< unsigned >( input_image->width() );
  auto const nj = static_cast< unsigned >( input_image->height() );

  // Return the previously-computed one if the size is the same
  if( spatial_prior.ni() == ni && spatial_prior.nj() == nj )
  {
    return spatial_prior;
  }

  spatial_prior = vil_image_view< vxl_byte >( ni, nj, 1 );

  double scale_factor =
    static_cast< double >( std::numeric_limits< vxl_byte >::max() ) /
    ( grid_length * grid_length - 1 );

  for( auto const i : kvr::iota( ni ) )
  {
    auto const i_id = ( grid_length * i ) / ni;
    for( auto const j : kvr::iota( nj ) )
    {
      auto const j_id = ( grid_length * j ) / nj;
      auto const index = grid_length * j_id + i_id;
      spatial_prior( i, j ) = static_cast< vxl_byte >( index * scale_factor );
    }
  }
  return spatial_prior;
}

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< pix_t >
pixel_feature_extractor::priv
::concatenate_images( std::vector< vil_image_view< pix_t > > filtered_images )
{
  // Count the total number of planes
  unsigned total_planes{ 0 };

  for( auto const& image : filtered_images )
  {
    total_planes += image.nplanes();
  }

  if( total_planes == 0 )
  {
    LOG_ERROR( p->logger(), "No filtered images provided" );
    return {};
  }

  auto const ni = filtered_images.at( 0 ).ni();
  auto const nj = filtered_images.at( 0 ).nj();
  vil_image_view< pix_t > concatenated_planes{ ni, nj, total_planes };

  // Concatenate the filtered images into a single output
  unsigned current_plane{ 0 };

  for( auto const& filtered_image : filtered_images )
  {
    for( unsigned i{ 0 }; i < filtered_image.nplanes(); ++i )
    {
      auto output_plane = vil_plane( concatenated_planes, current_plane );
      auto input_plane = vil_plane( filtered_image, i );
      output_plane.deep_copy( input_plane );
      ++current_plane;
    }
  }
  return concatenated_planes;
}

// ----------------------------------------------------------------------------
// Convert to a narrower type without wrapping
template < typename out_t, typename in_t >
vil_image_view< out_t >
clamping_cast( vil_image_view< in_t > input_image )
{
  // Safely compute the min and max given any type of inputs
  constexpr auto casted_in_min =
    static_cast< double >( std::numeric_limits< in_t >::min() );
  constexpr auto casted_out_min =
    static_cast< double >( std::numeric_limits< out_t >::min() );

  constexpr auto casted_in_max =
    static_cast< double >( std::numeric_limits< in_t >::max() );
  constexpr auto casted_out_max =
    static_cast< double >( std::numeric_limits< out_t >::max() );

  if( casted_in_min < casted_out_min || casted_in_max > casted_out_max )
  {
    constexpr auto min_output_value =
      ( casted_in_min > casted_out_min
        ? std::numeric_limits< in_t >::min()
        : static_cast< in_t >( std::numeric_limits< out_t >::min() ) );

    constexpr auto max_output_value =
      ( casted_in_max < casted_out_max
        ? std::numeric_limits< in_t >::max()
        : static_cast< in_t >( std::numeric_limits< out_t >::max() ) );

    vil_clamp( input_image, input_image, min_output_value, max_output_value );
  }

  vil_image_view< out_t > output_image;
  vil_convert_cast( input_image, output_image );
  return output_image;
}

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< pix_t >
convert_to_typed_vil_image_view(
  kwiver::vital::image_container_sptr input_image )
{
  auto const vxl_image_ptr = vxl::image_container::vital_to_vxl(
    input_image->get_image() );
  return vil_convert_cast( pix_t(), vxl_image_ptr );
}

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< pix_t >
pixel_feature_extractor::priv
::filter( kwiver::vital::image_container_sptr input_image )
{
  ++frame_number;

  std::vector< vil_image_view< pix_t > > filtered_images;

  vil_image_view< double > double_variance;

  if( enable_color )
  {
    // 3 channels
    filtered_images.push_back(
      convert_to_typed_vil_image_view< pix_t >( input_image ) );
  }

  // These three features require processing the vil_image directly
  if( enable_gray || enable_average || enable_normalized_variance )
  {
    auto input_image_sptr =
      vxl::image_container::vital_to_vxl( input_image->get_image() );

    if( input_image_sptr->nplanes() == 3 )
    {
      input_image_sptr = vil_convert_to_grey_using_rgb_weighting(
        input_image_sptr );
    }
    else
    {
      input_image_sptr = vil_convert_to_grey_using_average( input_image_sptr );
    }

    auto const double_gray = static_cast< vil_image_view< double > >(
      vil_convert_cast( double{}, input_image_sptr ) );

    if( enable_average || enable_normalized_variance )
    {
      auto gray_container =
        std::make_shared< vxl::image_container >( double_gray );
      double_variance = convert_to_typed_vil_image_view< double >(
        average_frames_filter->filter( gray_container ) );
    }

    // 1 channel
    if( enable_gray )
    {
      auto pix_t_gray = clamping_cast< pix_t >( double_gray );
      filtered_images.push_back( pix_t_gray );
    }
  }

  if( enable_color_commonality )
  {
    // 1 channel
    auto color_commonality = convert_to_typed_vil_image_view< pix_t >(
      color_commonality_filter->filter( input_image ) );

    filtered_images.push_back( color_commonality );
  }
  if( enable_high_pass_box )
  {
    auto high_pass_box = convert_to_typed_vil_image_view< pix_t >(
      high_pass_box_filter->filter( input_image ) );

    // Legacy BurnOut models expect these channels to be incorrectly ordered
    // TODO Remove this code when we no longer need to train models using
    // legacy code
    auto first_plane = vil_plane( high_pass_box, 0 );
    auto second_plane = vil_plane( high_pass_box, 1 );
    auto temp = vil_copy_deep( first_plane );
    first_plane.deep_copy( second_plane );
    second_plane.deep_copy( temp );

    // 3 channels
    filtered_images.push_back( high_pass_box );
  }
  if( enable_high_pass_bidir )
  {
    auto high_pass_bidir = convert_to_typed_vil_image_view< pix_t >(
      high_pass_bidir_filter->filter( input_image ) );
    // 3 channels
    filtered_images.push_back( high_pass_bidir );
  }

  // TODO consider naming this variance since that option is used more
  if( enable_average )
  {
    auto variance = clamping_cast< pix_t >( double_variance );

    // 1 channel
    filtered_images.push_back( variance );
  }
  if( enable_aligned_edge )
  {
    auto aligned_edge = convert_to_typed_vil_image_view< pix_t >(
      aligned_edge_detection_filter->filter( input_image ) );

    auto joint_response =
      vil_plane( aligned_edge, aligned_edge.nplanes() - 1 );
    // 1 channel
    filtered_images.push_back( joint_response );
  }
  if( enable_normalized_variance )
  {
    // Since variance is a double and may be small, avoid premptively casting
    // to a byte
    auto scale_factor =
      variance_scale_factor / static_cast< float >( frame_number );
    vil_math_scale_values( double_variance, scale_factor );

    auto variance = clamping_cast< pix_t >( double_variance );
    // 1 channel
    filtered_images.push_back( variance );
  }
  if( enable_spatial_prior )
  {
    auto spatial_prior = generate_spatial_prior( input_image );
    // 1 channel
    filtered_images.push_back( spatial_prior );
  }

  vil_image_view< pix_t > concatenated_out =
    concatenate_images< pix_t >( filtered_images );

  return concatenated_out;
}

// ----------------------------------------------------------------------------
pixel_feature_extractor
::pixel_feature_extractor()
  : d{ new priv{ this } }
{
  attach_logger( "arrows.vxl.pixel_feature_extractor" );
}

// ----------------------------------------------------------------------------
pixel_feature_extractor
::~pixel_feature_extractor()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
pixel_feature_extractor
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "enable_color",
                     d->enable_color,
                     "Enable color channels." );
  config->set_value( "enable_gray",
                     d->enable_gray,
                     "Enable grayscale channel." );
  config->set_value( "enable_aligned_edge",
                     d->enable_aligned_edge,
                     "Enable aligned_edge_detection filter." );
  config->set_value( "enable_average",
                     d->enable_average,
                     "Enable average_frames filter." );
  config->set_value( "enable_color_commonality",
                     d->enable_color_commonality,
                     "Enable color_commonality_filter filter." );
  config->set_value( "enable_high_pass_box",
                     d->enable_high_pass_box,
                     "Enable high_pass_filter filter." );
  config->set_value( "enable_high_pass_bidir",
                     d->enable_high_pass_bidir,
                     "Enable high_pass_filter filter." );
  config->set_value( "enable_normalized_variance",
                     d->enable_normalized_variance,
                     "Enable the normalized variance since the last shot break. "
                     "This will be a scalar multiple with the normal variance until "
                     "shot breaks are implemented." );
  config->set_value( "enable_spatial_prior",
                     d->enable_spatial_prior,
                     "Enable an image which encodes the location" );
  config->set_value( "variance_scale_factor",
                     d->variance_scale_factor,
                     "The multiplicative value for the normalized varaince" );
  config->set_value( "grid_length",
                     d->grid_length,
                     "The number of grids in each directions of the spatial "
                     "prior" );
  return config;
}

// ----------------------------------------------------------------------------
void
pixel_feature_extractor
::set_configuration( vital::config_block_sptr in_config )
{
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->enable_color = config->get_value< bool >( "enable_color" );
  d->enable_gray = config->get_value< bool >( "enable_gray" );
  d->enable_aligned_edge = config->get_value< bool >( "enable_aligned_edge" );
  d->enable_average = config->get_value< bool >( "enable_average" );
  d->enable_color_commonality = config->get_value< bool >(
    "enable_color_commonality" );
  d->enable_high_pass_box =
    config->get_value< bool >( "enable_high_pass_box" );
  d->enable_high_pass_bidir =
    config->get_value< bool >( "enable_high_pass_bidir" );
  d->enable_normalized_variance =
    config->get_value< bool >( "enable_normalized_variance" );
  d->enable_spatial_prior =
    config->get_value< bool >( "enable_spatial_prior" );

  d->variance_scale_factor =
    config->get_value< float >( "variance_scale_factor" );
  d->grid_length = config->get_value< unsigned >( "grid_length" );

  // Configure the individual filter algorithms
  d->aligned_edge_detection_filter->set_configuration(
    config->subblock_view( "aligned_edge" ) );
  d->average_frames_filter->set_configuration(
    config->subblock_view( "average" ) );
  d->color_commonality_filter->set_configuration(
    config->subblock_view( "color_commonality" ) );
  d->high_pass_box_filter->set_configuration(
    config->subblock_view( "high_pass_box" ) );
  d->high_pass_bidir_filter->set_configuration(
    config->subblock_view( "high_pass_bidir" ) );
}

// ----------------------------------------------------------------------------
bool
pixel_feature_extractor
::check_configuration( vital::config_block_sptr config ) const
{
  auto enable_color = config->get_value< bool >( "enable_color" );
  auto enable_gray = config->get_value< bool >( "enable_gray" );
  auto enable_aligned_edge =
    config->get_value< bool >( "enable_aligned_edge" );
  auto enable_average = config->get_value< bool >( "enable_average" );
  auto enable_color_commonality = config->get_value< bool >(
    "enable_color_commonality" );
  auto enable_high_pass_box =
    config->get_value< bool >( "enable_high_pass_box" );
  auto enable_high_pass_bidir =
    config->get_value< bool >( "enable_high_pass_bidir" );
  auto enable_normalized_variance =
    config->get_value< bool >( "enable_normalized_variance" );
  auto enable_spatial_prior =
    config->get_value< bool >( "enable_spatial_prior" );

  if( !( enable_color || enable_gray || enable_aligned_edge ||
         enable_average || enable_color_commonality || enable_high_pass_box ||
         enable_high_pass_bidir || enable_normalized_variance ||
         enable_spatial_prior ) )
  {
    LOG_ERROR( logger(), "At least one filter must be enabled" );
    return false;
  }

  return d->check_sub_algorithm( config, "aligned_edge" ) &&
         d->check_sub_algorithm( config, "average" ) &&
         d->check_sub_algorithm( config, "color_commonality" ) &&
         d->check_sub_algorithm( config, "high_pass_box" ) &&
         d->check_sub_algorithm( config, "high_pass_bidir" );
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
pixel_feature_extractor
::filter( kwiver::vital::image_container_sptr image )
{
  // Perform Basic Validation
  if( !image )
  {
    LOG_ERROR( logger(), "Invalid image" );
    return kwiver::vital::image_container_sptr();
  }

  // Filter and with responses cast to bytes
  auto const responses = d->filter< vxl_byte >( image );

  return std::make_shared< vxl::image_container >(
    vxl::image_container{ responses } );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
