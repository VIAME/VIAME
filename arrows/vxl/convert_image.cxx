/*ckwg +5
 * Copyright 2019 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "convert_image.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_image_view.h>
#include <vil/vil_convert.h>

#include <limits>
#include <type_traits>


namespace kwiver {
namespace arrows {
namespace vxl {

namespace
{

// Convert a floating point image to an intergral type by multiplying
// it by a scaling factor in addition to thresholding it in one operation.
// Performs rounding.
template< typename InType, typename OutType >
void scale_image( const vil_image_view<InType>& src,
                  vil_image_view<OutType>& dst,
                  const double& dp_scale )
{
  // Resize, cast and copy
  unsigned ni = src.ni(), nj = src.nj(), np = src.nplanes();
  dst.set_size( ni, nj, np );

  std::ptrdiff_t sistep=src.istep(), sjstep=src.jstep(), spstep=src.planestep();
  std::ptrdiff_t distep=dst.istep(), djstep=dst.jstep(), dpstep=dst.planestep();

  const OutType max_output_value = std::numeric_limits<OutType>::max();

  const InType max_input_value = static_cast< InType >(
    static_cast< double >( max_output_value ) / dp_scale );

  const InType scale = static_cast< InType >( dp_scale );

  const InType* splane = src.top_left_ptr();
  OutType* dplane = dst.top_left_ptr();
  for( unsigned p=0;p<np;++p,splane += spstep,dplane += dpstep )
  {
    const InType* srow = splane;
    OutType* drow = dplane;
    for( unsigned j=0;j<nj;++j,srow += sjstep,drow += djstep )
    {
      const InType* spixel = srow;
      OutType* dpixel = drow;
      for( unsigned i=0;i<ni;++i,spixel+=sistep,dpixel+=distep )
      {
        if( *spixel <= max_input_value )
        {
          *dpixel = static_cast<OutType>( *spixel * scale + 0.5 );
        }
        else
        {
          *dpixel = max_output_value;
        }
      }
    }
  }
}

} // end anonoymous namespace

// --------------------------------------------------------------------------------------
/// Private implementation class
class convert_image::priv
{
public:

  priv()
   : output_format( "byte" )
   , scale_factor( 0.0 )
  {
  }

  ~priv()
  {
  }

  std::string output_format;
  double scale_factor;  
};

// --------------------------------------------------------------------------------------
convert_image
::convert_image()
: d( new priv() )
{
  attach_logger( "arrows.vxl.convert_image" );
}

convert_image
::~convert_image()
{
}

vital::config_block_sptr
convert_image
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "output_format", d->output_format,
    "Output type format: byte, sbyte, float, double, uint16, uint32, etc." );
  config->set_value( "scale_factor", d->scale_factor,
    "Optional value scaling factor" );

  return config;
}

void
convert_image
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing a
  // get_value() call.
  vital::config_block_sptr config =
    this->get_configuration();
  config->merge_config( in_config );

  // Settings for conversion
  d->output_format = config->get_value< std::string >( "output_format" );
  d->scale_factor = config->get_value< double >( "scale_factor" );
}

bool
convert_image
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}

// Perform stitch operation
kwiver::vital::image_container_sptr
convert_image
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_OUTPUT_CASE(S,T)                                        \
  if( d->output_format == S )                                     \
  {                                                                    \
    typedef vil_pixel_format_type_of<T >::component_type opix_t;       \
    vil_image_view< opix_t > output;                                   \
    if( d->scale_factor == 0.0 || d->scale_factor == 1.0 )             \
    {                                                                  \
      vil_convert_cast( input, output );                               \
    }                                                                  \
    else                                                               \
    {                                                                  \
      scale_image<ipix_t,opix_t>( input, output, d->scale_factor );    \
    }                                                                  \
    return std::make_shared< vxl::image_container >( output );         \
  }                                                                    \

#define HANDLE_INPUT_CASE(T)                                           \
  case T:                                                              \
    {                                                                  \
      typedef vil_pixel_format_type_of<T >::component_type ipix_t;     \
      vil_image_view< ipix_t > input = view;                           \
      if( d->output_format == "disable" )                              \
      {                                                                \
        return image_data;                                             \
      }                                                                \
      HANDLE_OUTPUT_CASE("bool", VIL_PIXEL_FORMAT_BOOL);               \
      HANDLE_OUTPUT_CASE("byte", VIL_PIXEL_FORMAT_BYTE);               \
      HANDLE_OUTPUT_CASE("sbyte", VIL_PIXEL_FORMAT_SBYTE);             \
      HANDLE_OUTPUT_CASE("uint16", VIL_PIXEL_FORMAT_UINT_16);          \
      HANDLE_OUTPUT_CASE("int16", VIL_PIXEL_FORMAT_INT_16);            \
      HANDLE_OUTPUT_CASE("uint32", VIL_PIXEL_FORMAT_UINT_32);          \
      HANDLE_OUTPUT_CASE("int32", VIL_PIXEL_FORMAT_INT_32);            \
      HANDLE_OUTPUT_CASE("uint64", VIL_PIXEL_FORMAT_UINT_64);          \
      HANDLE_OUTPUT_CASE("int64", VIL_PIXEL_FORMAT_INT_64);            \
      HANDLE_OUTPUT_CASE("float", VIL_PIXEL_FORMAT_FLOAT);             \
      HANDLE_OUTPUT_CASE("double", VIL_PIXEL_FORMAT_DOUBLE);           \
    }                                                                  \
    break;                                                             \

  switch( view->pixel_format() )
  {
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_BOOL);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_SBYTE);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_16);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_16);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_32);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_32);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_64);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_64);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_FLOAT);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_DOUBLE);

  default:
    LOG_ERROR( logger(), "Invalid input format type received" );
    return kwiver::vital::image_container_sptr();
  }

#undef HANDLE_INPUT_CASE
#undef HANDLE_OUTPUT_CASE

  LOG_ERROR( logger(), "Invalid output format type received" );
  return kwiver::vital::image_container_sptr();
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
