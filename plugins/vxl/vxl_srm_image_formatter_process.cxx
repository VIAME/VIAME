/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Register multi-modal images.
 */

#include "vxl_srm_image_formatter_process.h"

#include <vital/vital_types.h>

#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/homography.h>

#include <arrows/vxl/image_container.h>

#include <vector>

#include <vil/vil_image_view.h>
#include <vil/vil_copy.h>
#include <vil/vil_crop.h>
#include <vil/vil_fill.h>
#include <vil/vil_plane.h>
#include <vil/vil_resample_bilin.h>


namespace viame
{

namespace vxl
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
class vxl_srm_image_formatter_process::priv
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
  void filter( const vil_image_view< PixType >& input,
    std::vector< vil_image_view< PixType > >& output );
};

// =============================================================================

vxl_srm_image_formatter_process
::vxl_srm_image_formatter_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new vxl_srm_image_formatter_process::priv() )
{
  make_ports();
  make_config();
}


vxl_srm_image_formatter_process
::~vxl_srm_image_formatter_process()
{
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
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
void vxl_srm_image_formatter_process::priv
::filter( const vil_image_view< PixType >& raw_input,
          std::vector< vil_image_view< PixType > >& output )
{
  typedef vil_image_view< PixType > image_t;

  // Verification of input
  if( raw_input.ni() == 0 || raw_input.nj() == 0 )
  {
    output.push_back( image_t() );
    return;
  }

  // Update recorded image properties
  m_max_input_width = std::max( raw_input.ni(), m_max_input_width );
  m_max_input_height = std::max( raw_input.nj(), m_max_input_height );

  // Confirm not RGBA
  image_t input = raw_input;

  if( raw_input.nplanes() == 4 )
  {
    input = vil_image_view< PixType >( raw_input.ni(), raw_input.nj(), 3 );

    image_t tmp1 = vil_plane( input, 0 );
    image_t tmp2 = vil_plane( input, 1 );
    image_t tmp3 = vil_plane( input, 2 );

    vil_copy_reformat( vil_plane( raw_input, 0 ), tmp1 );
    vil_copy_reformat( vil_plane( raw_input, 1 ), tmp2 );
    vil_copy_reformat( vil_plane( raw_input, 2 ), tmp3 );
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
      output_ni = input.ni();
      output_nj = input.nj();

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

    if( input.ni() == output_ni && input.nj() == output_nj )
    {
      output.push_back( input );
      return;
    }

    vil_image_view< PixType > scaled;
    vil_resample_bilin( input, scaled, output_ni, output_nj );

    output.push_back( scaled );
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
      output_ni = std::min( m_max_output_width, input.ni() );
      output_nj = std::min( m_max_output_height, input.nj() );

      if( !m_first_output_width )
      {
        m_first_output_width = output_ni;
        m_first_output_height = output_nj;
      }
    }

    if( input.ni() == output_ni && input.nj() == output_nj )
    {
      output.push_back( input );
      return;
    }

    if( output_ni > input.ni() || output_nj > input.nj() )
    {
      vil_image_view< PixType > padded_crop( output_ni, output_nj, input.nplanes() );
      vil_fill( padded_crop, static_cast< PixType >( 0 ) );

      unsigned copy_ni = std::min( output_ni, input.ni() );
      unsigned copy_nj = std::min( output_nj, input.nj() );

      vil_image_view< PixType > dest =
        vil_crop( padded_crop, 0, copy_ni, 0, copy_nj );

      vil_copy_reformat( vil_crop( input, 0, copy_ni, 0, copy_nj ), dest );
      output.push_back( padded_crop );
    }
    else
    {
      output.push_back( vil_crop( input, 0, output_ni, 0, output_nj ) );
    }
  }
  else // Chip mode
  {
    // NOT YET IMPLEMENTATED
  }
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
::_step()
{
  kwiver::vital::image_container_sptr input_image =
    grab_from_port_using_trait( image );

  vil_image_view_base_sptr view =
    kwiver::arrows::vxl::image_container::vital_to_vxl(
      input_image->get_image() );

  // Perform different actions based on input type
#define HANDLE_CASE(T)                                                    \
  case T:                                                                 \
    {                                                                     \
      typedef vil_pixel_format_type_of<T >::component_type pix_t;         \
      vil_image_view< pix_t > input = view;                               \
                                                                          \
      std::vector< vil_image_view< pix_t > > outputs;                     \
      d->filter( input, outputs );                                        \
                                                                          \
      for( auto output : outputs )                                        \
      {                                                                   \
        if( output )                                                      \
        {                                                                 \
          push_to_port_using_trait( image,                                \
            std::make_shared< kwiver::arrows::vxl::image_container >(     \
              output ) );                                                 \
        }                                                                 \
        else                                                              \
        {                                                                 \
          push_to_port_using_trait( image,                                \
            kwiver::vital::image_container_sptr() );                      \
        }                                                                 \
      }                                                                   \
    }                                                                     \
    break;                                                                \

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
#undef HANDLE_CASE

  default:
    throw std::runtime_error( "Invalid type received" );
  }
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
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
vxl_srm_image_formatter_process
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
vxl_srm_image_formatter_process::priv
::priv()
  : m_max_input_width( 0 )
  , m_max_input_height( 0 )
  , m_first_output_width( 0 )
  , m_first_output_height( 0 )
{
}


vxl_srm_image_formatter_process::priv
::~priv()
{
}


} // end namespace vxl

} // end namespace viame
