// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "aligned_edge_detection.h"

#include <arrows/vxl/image_container.h>

#include <vil/algo/vil_gauss_filter.h>
#include <vil/algo/vil_sobel_3x3.h>
#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>

#include <limits>
#include <type_traits>

#include <cstdlib>

namespace kwiver {

namespace arrows {

namespace vxl {

// ----------------------------------------------------------------------------
/// Private implementation class
class aligned_edge_detection::priv
{
public:
  priv( aligned_edge_detection* parent ) : p{ parent } {}

  template < typename PixType, typename GradientType >
  vil_image_view< PixType >
  calculate_aligned_edges( vil_image_view< PixType > const& input,
                           vil_image_view< GradientType >& grad_i,
                           vil_image_view< GradientType >& grad_j );

  template < typename InputType, typename OutputType >
  vil_image_view< OutputType >
  nonmax_suppression( vil_image_view< InputType > const& grad_i,
                      vil_image_view< InputType > const& grad_j );

  template < typename pix_t > vil_image_view< pix_t >
  filter( vil_image_view< pix_t > const& input_image );

  aligned_edge_detection* p;
  // Internal parameters/settings
  float threshold = 10;
  bool produce_joint_output = true;
  double smoothing_sigma = 1.3;
  unsigned smoothing_half_step = 2;
};

// ----------------------------------------------------------------------------
// Perform NMS on the input gradient images in horizontal and vert directions
// only
template < typename InputType, typename OutputType >
vil_image_view< OutputType >
aligned_edge_detection::priv
::nonmax_suppression( vil_image_view< InputType > const& grad_i,
                      vil_image_view< InputType > const& grad_j )
{
  if( grad_i.ni() != grad_j.ni() || grad_i.nj() != grad_j.nj() )
  {
    LOG_ERROR(
      p->logger(),
      "Input gradient image dimensions must be equivalent" );
  }

  auto const ni = grad_i.ni();
  auto const nj = grad_i.nj();

  vil_image_view< OutputType > output( ni, nj, 2 );
  output.fill( 0 );

  // Perform non-maximum suppression
  for( decltype(+nj) j = 1; j < nj - 1; j++ )
  {
    for( decltype(+ni) i = 1; i < ni - 1; i++ )
    {
      const InputType val_i = grad_i( i, j );
      const InputType val_j = grad_j( i, j );

      if( val_i > threshold )
      {
        if( val_i >= grad_i( i - 1, j ) && val_i >= grad_i( i + 1, j ) )
        {
          output( i, j, 0 ) = static_cast< OutputType >( val_i );
        }
      }
      if( val_j > threshold )
      {
        if( val_j >= grad_j( i, j - 1 ) && val_j >= grad_j( i, j + 1 ) )
        {
          output( i, j, 1 ) = static_cast< OutputType >( val_j );
        }
      }
    }
  }
  return output;
}

// ----------------------------------------------------------------------------
// Calculate potential edges
template < typename PixType, typename GradientType >
vil_image_view< PixType >
aligned_edge_detection::priv
::calculate_aligned_edges( vil_image_view< PixType > const& input,
                           vil_image_view< GradientType >& grad_i,
                           vil_image_view< GradientType >& grad_j )
{
  // Calculate sobel approx in x/y directions
  vil_sobel_3x3( input, grad_i, grad_j );

  // Take absolute value of gradients
  vil_transform< GradientType, GradientType( GradientType ) >(
    grad_i, std::abs );
  vil_transform< GradientType, GradientType( GradientType ) >(
    grad_j, std::abs );

  // Perform NMS in vertical/horizonal directions and threshold magnitude
  vil_image_view< PixType > output =
    nonmax_suppression< GradientType, PixType >( grad_i, grad_j );
  return output;
}

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< pix_t >
aligned_edge_detection::priv
::filter( vil_image_view< pix_t > const& input_image )
{
  auto const source_ni = input_image.ni();
  auto const source_nj = input_image.nj();

  vil_image_view< float > grad_i{ source_ni, source_nj };
  vil_image_view< float > grad_j{ source_ni, source_nj };

  vil_image_view< pix_t > aligned_edges =
    calculate_aligned_edges< pix_t, float >( input_image, grad_i, grad_j );

  // Perform extra op if enabled
  if( produce_joint_output )
  {
    vil_image_view< pix_t > combined_edges{ source_ni, source_nj };
    combined_edges.fill( 0 );

    // Add vertical and horizontal edge planes together and smooth
    vil_image_view< pix_t > joint_nms_edges;

    vil_math_image_sum( vil_plane( aligned_edges, 0 ),
                        vil_plane( aligned_edges, 1 ),
                        joint_nms_edges );

    unsigned half_step = smoothing_half_step;
    unsigned min_dim = std::min( joint_nms_edges.ni(), joint_nms_edges.nj() );

    if( 2 * half_step + 1 >= min_dim )
    {
      half_step = ( min_dim - 1 ) / 2;
    }

    if( half_step != 0 )
    {
      vil_gauss_filter_2d( joint_nms_edges, combined_edges, smoothing_sigma,
                           half_step );
    }
    else
    {
      vil_copy_reformat( joint_nms_edges, combined_edges );
    }
    vil_image_view< pix_t > all_channels( joint_nms_edges.ni(),
                                          joint_nms_edges.nj(), 3 );
    vil_plane( all_channels, 0 ).deep_copy( vil_plane( aligned_edges, 0 ) );
    vil_plane( all_channels, 1 ).deep_copy( vil_plane( aligned_edges, 1 ) );
    vil_plane( all_channels, 2 ).deep_copy( joint_nms_edges );
    return all_channels;
  }
  return aligned_edges;
}

// ----------------------------------------------------------------------------
aligned_edge_detection
::aligned_edge_detection()
  : d{ new priv{ this } }
{
  attach_logger( "arrows.vxl.aligned_edge_detection" );
}

// ----------------------------------------------------------------------------
aligned_edge_detection
::~aligned_edge_detection()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
aligned_edge_detection
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "threshold",
                     d->threshold,
                     "Minimum edge magnitude required to report as an edge "
                     "in any output image." );
  config->set_value( "produce_joint_output",
                     d->produce_joint_output,
                     "Set to false if we do not want to spend time computing "
                     "joint edge images comprised of both horizontal and "
                     "vertical information." );
  config->set_value( "smoothing_sigma",
                     d->smoothing_sigma,
                     "Smoothing sigma for the output NMS edge density map." );
  config->set_value( "smoothing_half_step",
                     d->smoothing_half_step,
                     "Smoothing half step for the output NMS edge density "
                     "map." );

  return config;
}

// ----------------------------------------------------------------------------
void
aligned_edge_detection
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values are present. An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for edge detection
  d->threshold = config->get_value< float >( "threshold" );
  d->produce_joint_output =
    config->get_value< bool >( "produce_joint_output" );
  d->smoothing_sigma = config->get_value< double >( "smoothing_sigma" );
  d->smoothing_half_step =
    config->get_value< unsigned >( "smoothing_half_step" );
}

// ----------------------------------------------------------------------------
bool
aligned_edge_detection
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
aligned_edge_detection
::filter( kwiver::vital::image_container_sptr source_image_ptr )
{
  if( !source_image_ptr )
  {
    LOG_ERROR( logger(), "Image pointer was not valid." );
    return nullptr;
  }

  // Get input image
  vil_image_view_base_sptr source_image =
    vxl::image_container::vital_to_vxl( source_image_ptr->get_image() );

  // Perform Basic Validation
  if( !source_image )
  {
    LOG_ERROR( logger(), "Image was not valid." );
    return nullptr;
  }

  // Perform Basic Validation
  if( source_image->nplanes() == 3 )
  {
    source_image = vil_convert_to_grey_using_average( source_image );
  }
  else if( source_image->nplanes() != 1 )
  {
    LOG_ERROR( logger(), "Input must be a grayscale image!" );
    return nullptr;
  }

#define HANDLE_CASE( T )                                          \
  case T:                                                         \
  {                                                               \
    using ipix_t = vil_pixel_format_type_of< T >::component_type; \
    auto filtered = d->filter< ipix_t >( source_image );          \
    auto container = vxl::image_container{ filtered };            \
    return std::make_shared< vxl::image_container >( container ); \
  }                                                               \

  switch( source_image->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_FLOAT );
#undef HANDLE_CASE
    default:
      LOG_ERROR( logger(), "Invalid input format type received" );
      return nullptr;
  }

  LOG_ERROR( logger(), "Invalid output format type received" );
  return nullptr;
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
