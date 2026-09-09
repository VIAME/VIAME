/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Register multi-modal images.
 */

#include "format_images_srm_process.h"

#include <vital/vital_types.h>

#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/homography.h>

#include <image_ops/channels.h>
#include <image_ops/dispatch.h>
#include <image_ops/resample.h>

#include <algorithm>
#include <vector>


namespace viame
{

namespace descriptors
{

create_config_trait( fix_output_size, bool, "true",
  "Should the output image size always be consistent and unchanging" );

create_config_trait( resize_option, std::string, "rescale",
  "Option to meet output size parameter, can be: rescale, chip, or crop." );

create_config_trait( max_output_width, unsigned, "10000",
  "Maximum allowed image width of archive after a potential resize" );
create_config_trait( max_output_height, unsigned, "10000",
  "Maximum allowed image height of archive after a potential resize" );

create_config_trait( chip_overlap, unsigned, "50",
  "If we're chipping a large image into smaller chips, this is the approximate "
  "overlap between neighboring chips in terms of pixels." );
create_config_trait( pad_sides, bool, "false",
  "If the computed image does not match the max output size, pad it with "
  "black pixels to meet the size." );
create_config_trait( flux_factor, double, "0.05",
  "Allowable error for resizing images to meet a more desirable size." );
create_config_trait( max_images_per_index, unsigned, "-1",
  "Maximum number of images that can be stored together in the same "
  "database index." );


//------------------------------------------------------------------------------
// Private implementation class
class format_images_srm_process::priv
{
public:
  priv();
  ~priv();

  // Configuration parameters
  bool m_fix_output_size;

  enum{ RESCALE, CHIP, CROP } m_resize_option;

  unsigned m_max_output_width;
  unsigned m_max_output_height;

  unsigned m_chip_overlap;
  bool m_pad_sides;
  double m_flux_factor;
  unsigned m_max_images_per_index;

  // Computed parameters
  unsigned m_max_input_width;
  unsigned m_max_input_height;

  unsigned m_first_output_width;
  unsigned m_first_output_height;

  // Functions
  template< typename PixType >
  void filter( const kwiver::vital::image_of< PixType >& input,
    std::vector< kwiver::vital::image_of< PixType > >& output );
};

// =============================================================================

format_images_srm_process
::format_images_srm_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new format_images_srm_process::priv() )
{
  make_ports();
  make_config();
}


format_images_srm_process
::~format_images_srm_process()
{
}


// -----------------------------------------------------------------------------
void
format_images_srm_process
::_configure()
{
  d->m_fix_output_size =
    config_value_using_trait( fix_output_size );
  d->m_max_output_width =
    config_value_using_trait( max_output_width );
  d->m_max_output_height =
    config_value_using_trait( max_output_height );
  d->m_chip_overlap =
    config_value_using_trait( chip_overlap );
  d->m_pad_sides =
    config_value_using_trait( pad_sides );
  d->m_flux_factor =
    config_value_using_trait( flux_factor );
  d->m_max_images_per_index =
    config_value_using_trait( max_images_per_index );

  std::string mode = config_value_using_trait( resize_option );

  if( mode == "rescale" )
  {
    d->m_resize_option = priv::RESCALE;
  }
  else if( mode == "chip" )
  {
    d->m_resize_option = priv::CHIP;
  }
  else if( mode == "crop" )
  {
    d->m_resize_option = priv::CROP;
  }
  else
  {
    throw std::runtime_error( "Invalid resize option: " + mode );
  }
}


// -----------------------------------------------------------------------------
template< typename PixType >
void format_images_srm_process::priv
::filter( const kwiver::vital::image_of< PixType >& raw_input,
          std::vector< kwiver::vital::image_of< PixType > >& output )
{
  typedef kwiver::vital::image_of< PixType > image_t;
  namespace io = viame::image_ops;

  // Verification of input
  if( raw_input.width() == 0 || raw_input.height() == 0 )
  {
    output.push_back( image_t() );
    return;
  }

  // Update recorded image properties
  m_max_input_width =
    std::max( static_cast< unsigned >( raw_input.width() ), m_max_input_width );
  m_max_input_height =
    std::max( static_cast< unsigned >( raw_input.height() ), m_max_input_height );

  // Confirm not RGBA
  image_t input = raw_input;

  if( raw_input.depth() == 4 )
  {
    // Drop the alpha rather than compositing it, as before
    input = io::force_three_channels( raw_input );
  }

  // Handle correct case
  if( m_resize_option == RESCALE )
  {
    unsigned output_ni;
    unsigned output_nj;

    if( m_fix_output_size && m_first_output_width )
    {
      output_ni = m_first_output_width;
      output_nj = m_first_output_height;
    }
    else
    {
      output_ni = input.width();
      output_nj = input.height();

      if( output_ni > m_max_output_width || output_nj > m_max_output_height )
      {
        double scale_factor = std::min(
          static_cast< double >( m_max_output_width ) / output_ni,
          static_cast< double >( m_max_output_height ) / output_nj );

        output_ni = static_cast< unsigned >( scale_factor * output_ni );
        output_nj = static_cast< unsigned >( scale_factor * output_nj );
      }

      if( !m_first_output_width )
      {
        m_first_output_width = output_ni;
        m_first_output_height = output_nj;
      }
    }

    if( input.width() == output_ni && input.height() == output_nj )
    {
      output.push_back( input );
      return;
    }

    output.push_back( io::resize_bilinear( input, output_ni, output_nj ) );
  }
  else if( m_resize_option == CROP )
  {
    unsigned output_ni;
    unsigned output_nj;

    if( m_fix_output_size && m_first_output_width )
    {
      output_ni = m_first_output_width;
      output_nj = m_first_output_height;
    }
    else
    {
      output_ni = std::min( m_max_output_width,
                            static_cast< unsigned >( input.width() ) );
      output_nj = std::min( m_max_output_height,
                            static_cast< unsigned >( input.height() ) );

      if( !m_first_output_width )
      {
        m_first_output_width = output_ni;
        m_first_output_height = output_nj;
      }
    }

    if( input.width() == output_ni && input.height() == output_nj )
    {
      output.push_back( input );
      return;
    }

    if( output_ni > input.width() || output_nj > input.height() )
    {
      output.push_back( io::pad_or_crop( input, output_ni, output_nj ) );
    }
    else
    {
      output.push_back( io::crop( input, 0, 0, output_ni, output_nj ) );
    }
  }
  else // Chip mode
  {
    // NOT YET IMPLEMENTATED
  }
}


// -----------------------------------------------------------------------------
void
format_images_srm_process
::_step()
{
  kwiver::vital::image_container_sptr input_image =
    grab_from_port_using_trait( image );

  namespace io = viame::image_ops;

  std::vector< kwiver::vital::image_container_sptr > results;

  io::dispatch_pixel_type(
    input_image->get_image(),
    [ & ]( auto const& typed ) -> int
    {
      using pix_t = io::pixel_type_t< decltype( typed ) >;

      std::vector< kwiver::vital::image_of< pix_t > > outputs;
      d->filter( typed, outputs );

      for( auto const& output : outputs )
      {
        results.push_back(
          output.width() > 0 && output.height() > 0
          ? std::make_shared< kwiver::vital::simple_image_container >( output )
          : kwiver::vital::image_container_sptr() );
      }

      return 0;
    } );

  for( auto const& result : results )
  {
    push_to_port_using_trait( image, result );
  }
}


// -----------------------------------------------------------------------------
void
format_images_srm_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( timestamp, optional );

  // -- output --
  declare_output_port_using_trait( image, optional );
  declare_output_port_using_trait( timestamp, optional );
}


// -----------------------------------------------------------------------------
void
format_images_srm_process
::make_config()
{
  declare_config_using_trait( fix_output_size );
  declare_config_using_trait( max_output_width );
  declare_config_using_trait( max_output_height );
  declare_config_using_trait( resize_option );
  declare_config_using_trait( chip_overlap );
  declare_config_using_trait( pad_sides );
  declare_config_using_trait( flux_factor );
  declare_config_using_trait( max_images_per_index );
}


// =============================================================================
format_images_srm_process::priv
::priv()
  : m_max_input_width( 0 )
  , m_max_input_height( 0 )
  , m_first_output_width( 0 )
  , m_first_output_height( 0 )
{
}


format_images_srm_process::priv
::~priv()
{
}


} // end namespace descriptors

} // end namespace viame
