/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "perform_white_balancing.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_pixel_format.h>
#include <vil/vil_copy.h>

#include <exception>

namespace viame {

namespace kv = kwiver::vital;
namespace vxl = kwiver::arrows::vxl;


// --------------------------------------------------------------------------------------
/// Private implementation class
class perform_white_balancing::priv
{
public:

  priv() : m_settings()
  {
  }

  ~priv()
  {
  }

  // Internal parameters/settings
  std::unique_ptr< auto_white_balancer_base > m_balancer;
  auto_white_balancer_settings m_settings;
};

// --------------------------------------------------------------------------------------
perform_white_balancing
::perform_white_balancing()
: d( new priv() )
{
  attach_logger( "viame.vxl.perform_white_balancing" );
}

perform_white_balancing
::~perform_white_balancing()
{
}

kv::config_block_sptr
perform_white_balancing
::get_configuration() const
{
  // get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "white_scale_factor",
    d->m_settings.white_traverse_factor,
    "A measure of how much to over or under correct white "
    "reference points. A value near 1.0 will attempt to make "
    "the whitest thing in the image very close to pure white." );
  config->set_value( "black_scale_factor",
    d->m_settings.black_traverse_factor,
    "A measure of how much to over or under correct black "
    "reference points. A value near 1.0 will attempt to make "
    "the blackest thing in the image very close to pure black." );
  config->set_value( "exp_history_factor",
    d->m_settings.exp_averaging_factor,
    "The exponential averaging factor for correction matrices" );
  config->set_value( "matrix_resolution",
    d->m_settings.correction_matrix_res,
    "The resolution of the correction matrix" );

  return config;
}

void
perform_white_balancing
::set_configuration( kv::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing a
  // get_value() call.
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for averaging
  d->m_settings.white_traverse_factor =
    config->get_value< double >( "white_scale_factor" );
  d->m_settings.black_traverse_factor =
    config->get_value< double >( "black_scale_factor" );
  d->m_settings.exp_averaging_factor  =
    config->get_value< double >( "exp_history_factor" );
  d->m_settings.correction_matrix_res =
    config->get_value< unsigned> ( "matrix_resolution"  );

  // Validate parameters
  if( d->m_settings.exp_averaging_factor > 1.0 )
  {
    throw std::runtime_error( "Invalid exponential averaging weight" );
  }
  if( d->m_settings.correction_matrix_res > 200 )
  {
    throw std::runtime_error( "Correction matrix resolution is too large!" );
  }
}

bool
perform_white_balancing
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// Perform stitch operation
kv::image_container_sptr
perform_white_balancing
::filter( kv::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return image_data;
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_CASE(T)                                                          \
  case T:                                                                       \
    {                                                                           \
      typedef vil_pixel_format_type_of<T >::component_type pix_t;               \
      vil_image_view< pix_t > input = view;                                     \
                                                                                \
      auto_white_balancer< pix_t >* balancer =                                  \
        dynamic_cast< auto_white_balancer< pix_t >* >( d->m_balancer.get() );   \
                                                                                \
      if( !balancer )                                                           \
      {                                                                         \
        balancer = new auto_white_balancer< pix_t >();                          \
        balancer->configure( d->m_settings );                                   \
        d->m_balancer = std::unique_ptr< auto_white_balancer_base >( balancer );\
      }                                                                         \
                                                                                \
      vil_image_view< pix_t > output;                                           \
      vil_copy_deep( input, output );                                           \
      balancer->apply( output );                                                \
      return std::make_shared< vxl::image_container >( output );                \
    }                                                                           \
    break;                                                                      \

  switch( view->pixel_format() )
  {
    HANDLE_CASE(VIL_PIXEL_FORMAT_BYTE);
    HANDLE_CASE(VIL_PIXEL_FORMAT_UINT_16);
    HANDLE_CASE(VIL_PIXEL_FORMAT_FLOAT);
    HANDLE_CASE(VIL_PIXEL_FORMAT_DOUBLE);
#undef HANDLE_CASE

  default:
    throw std::runtime_error( "Unsupported type received" );
  }

  // Code not reached, prevent warning
  return kv::image_container_sptr();
}

} // end namespace viame
