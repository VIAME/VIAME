// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "threshold.h"

#include "image_statistics.h"

#include <arrows/vxl/image_container.h>

#include <vital/util/enum_converter.h>

#include <vital/range/iota.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>

#include <limits>
#include <random>
#include <type_traits>

#include <cstdlib>

namespace kwiver {

namespace arrows {

namespace vxl {

enum threshold_mode
{
  MODE_absolute,
  MODE_percentile,
};

ENUM_CONVERTER( mode_converter, threshold_mode, { "absolute", MODE_absolute },
                { "percentile", MODE_percentile } );

// ----------------------------------------------------------------------------
// Private implementation class
class threshold::priv
{
public:
  template < typename pix_t >
  vil_image_view< bool >
  filter( vil_image_view< pix_t > image );

  double threshold{ 0.95 };
  threshold_mode type{ MODE_percentile };
};

// ----------------------------------------------------------------------------
template < typename pix_t >
vil_image_view< bool >
threshold::priv
::filter( vil_image_view< pix_t > image )
{
  switch( type )
  {
    case MODE_absolute:
    {
      vil_image_view< bool > output;
      vil_threshold_above( image, output, static_cast< pix_t >( threshold ) );
      return output;
    }
    case MODE_percentile:
    {
      vil_image_view< bool > output;
      percentile_threshold_above( image, threshold, output );
      return output;
    }
    default:
      return {};
  }
}

// ----------------------------------------------------------------------------
threshold
::threshold()
  : d{ new priv{} }
{
  attach_logger( "arrows.vxl.threshold" );
}

// ----------------------------------------------------------------------------
threshold
::~threshold()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
threshold
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "threshold", d->threshold,
                     "Threshold to use. Meaning is dependent on type." );
  config->set_value( "type", mode_converter().to_string( d->type ),
                     "Type of thresholding to use. Possible options are: " +
                     mode_converter().element_name_string() );

  return config;
}

// ----------------------------------------------------------------------------
void
threshold
::set_configuration( vital::config_block_sptr in_config )
{
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->threshold = config->get_value< double >( "threshold" );
  d->type = config->get_enum_value< mode_converter >( "type" );
}

// ----------------------------------------------------------------------------
bool
threshold
::check_configuration( vital::config_block_sptr in_config ) const
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  auto const threshold = config->get_value< double >( "threshold" );
  auto const type = config->get_enum_value< mode_converter >( "type" );
  if( type == MODE_percentile && ( threshold < 0.0 || threshold > 1.0 ) )
  {
    LOG_ERROR( logger(), "threshold must be in [0, 1] but instead was "
               << threshold );
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
threshold
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid image data." );
    return nullptr;
  }

  vil_image_view_base_sptr view{
    vxl::image_container::vital_to_vxl( image_data->get_image() ) };

#define HANDLE_CASE( T )                                               \
  case T:                                                              \
  {                                                                    \
    using ipix_t = vil_pixel_format_type_of< T >::component_type;      \
    vil_image_view< bool > thresholded{ d->filter< ipix_t >( view ) }; \
    return std::make_shared< vxl::image_container >( thresholded );    \
  }

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_FLOAT );
    HANDLE_CASE( VIL_PIXEL_FORMAT_DOUBLE );
#undef HANDLE_CASE

    default:
      LOG_ERROR( logger(), "Unsuported pixel type" );
      return nullptr;
  }
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
