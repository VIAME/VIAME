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

// ----------------------------------------------------------------------------
// Private implementation class
class pixel_feature_extractor::priv
{
public:

  priv( pixel_feature_extractor* parent ) : p{ parent }
  {
  }

  // Copy multiple filtered images into contigious memory
  template < typename pix_t > vil_image_view< pix_t >
  concatenate_images( std::vector< vil_image_view< pix_t > > filtered_images );
  // Extract local pixel-wise features
  template < typename response_t > vil_image_view< response_t >
  filter( kwiver::vital::image_container_sptr input_image );

  pixel_feature_extractor* p;

  bool enable_color{ true };
  bool enable_gray{ true };
  bool enable_aligned_edge{ true };
  bool enable_average{ true };
  bool enable_convert{ true };
  bool enable_color_commonality{ true };
  bool enable_high_pass_bidir{ true };
  bool enable_high_pass_box{ true };

  vxl::aligned_edge_detection aligned_edge_detection_filter;
  vxl::average_frames average_frames_filter;
  vxl::convert_image convert_image_filter;
  vxl::color_commonality_filter color_commonality_filter;
  vxl::high_pass_filter high_pass_bidir_filter;
  vxl::high_pass_filter high_pass_box_filter;
};

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

  for( auto const& image : filtered_images )
  {
    for( unsigned i{ 0 }; i < image.nplanes(); ++i )
    {
      vil_plane( concatenated_planes,
                 current_plane ).deep_copy( vil_plane( image, i ) );
      ++current_plane;
    }
  }
  return concatenated_planes;
}

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< pix_t >
pixel_feature_extractor::priv
::filter( kwiver::vital::image_container_sptr input_image )
{
  std::vector< vil_image_view< vxl_byte > > filtered_images;

  if( enable_color || enable_gray )
  {
    const auto vxl_image = vxl::image_container::vital_to_vxl( input_image->get_image() );
    // 3 channels
    if( enable_color ){
      filtered_images.push_back( vxl_image );
    }

    // 1 channel
    if( enable_gray )
    {
      // TODO check whether this should be vil_convert_to_grey_using_rgb_weighting
      filtered_images.push_back( vil_convert_to_grey_using_average( vxl_image ) );
    }
  }

  if( enable_color_commonality )
  {
    // 1 channel
    auto color_commonality = color_commonality_filter.filter( input_image );
    filtered_images.push_back(
        vxl::image_container::vital_to_vxl( color_commonality->get_image() ) );
  }
  // TODO this should also have one which does the other sort of filtering
  if( enable_high_pass_box )
  {
    auto high_pass_box = high_pass_box_filter.filter( input_image );
    // 2 channels
    filtered_images.push_back(
      vxl::image_container::vital_to_vxl( high_pass_box->get_image() ) );
  }
  if( enable_high_pass_bidir )
  {
    auto high_pass_bidir = high_pass_bidir_filter.filter( input_image );
    // 2 channels
    filtered_images.push_back(
      vxl::image_container::vital_to_vxl( high_pass_bidir->get_image() ) );
  }
  if( enable_convert )
  {
    // 3 channels
    // This should be removed
    auto converted = convert_image_filter.filter( input_image );
    filtered_images.push_back(
        vxl::image_container::vital_to_vxl( converted->get_image() ) );
  }
  // TODO this should be the variance instead
  if( enable_average )
  {
    // This should
    // Variance should be 1 channel
    auto averaged = average_frames_filter.filter( input_image );
    filtered_images.push_back(
        vxl::image_container::vital_to_vxl( averaged->get_image() ) );
  }
  if( enable_aligned_edge ){
    auto aligned_edge = aligned_edge_detection_filter.filter( input_image );
    // 2 channels
    filtered_images.push_back(
        vxl::image_container::vital_to_vxl( aligned_edge->get_image() ) );
  }

  vil_image_view< vxl_byte > concatenated_out =
    concatenate_images< vxl_byte >( filtered_images );

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
  config->set_value( "enable_convert",
                     d->enable_convert,
                     "Enable convert_image filter." );
  config->set_value( "enable_color_commonality",
                     d->enable_color_commonality,
                     "Enable color_commonality_filter filter." );
  config->set_value( "enable_high_pass_box",
                     d->enable_high_pass_box,
                     "Enable high_pass_filter filter." );
  config->set_value( "enable_high_pass_bidir",
                     d->enable_high_pass_bidir,
                     "Enable high_pass_filter filter." );
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
  d->enable_convert = config->get_value< bool >( "enable_convert" );
  d->enable_color_commonality = config->get_value< bool >(
    "enable_color_commonality" );
  d->enable_high_pass_box =
    config->get_value< bool >( "enable_high_pass_box" );
  d->enable_high_pass_bidir =
    config->get_value< bool >( "enable_high_pass_bidir" );

  // Configure the individual filter algorithms
  d->aligned_edge_detection_filter.set_configuration(
    config->subblock_view( "aligned_edge" ) );
  d->average_frames_filter.set_configuration(
     config->subblock_view( "average" ) );
  d->convert_image_filter.set_configuration(
    config->subblock_view( "convert" ) );
  d->color_commonality_filter.set_configuration(
    config->subblock_view( "color_commonality" ) );
  d->high_pass_box_filter.set_configuration(
    config->subblock_view( "high_pass_box" ) );
  d->high_pass_bidir_filter.set_configuration(
    config->subblock_view( "high_pass_bidir" ) );
  write_config_file( config->subblock_view( "high_pass_box" ), "box.conf" );
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
  auto enable_convert = config->get_value< bool >( "enable_convert" );
  auto enable_color_commonality = config->get_value< bool >(
    "enable_color_commonality" );
  auto enable_high_pass_box =
    config->get_value< bool >( "enable_high_pass_box" );
  auto enable_high_pass_bidir =
    config->get_value< bool >( "enable_high_pass_bidir" );

  if( !( enable_color || enable_gray || enable_aligned_edge ||
         enable_average || enable_convert || enable_color_commonality ||
         enable_high_pass_box || enable_high_pass_bidir) )
  {
    LOG_ERROR( logger(), "At least one filter must be enabled" );
    return false;
  }
  return true;
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
