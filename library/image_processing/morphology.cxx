/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "morphology.h"

#include <image_ops/dispatch.h>
#include <image_ops/morphology.h>

#include <viame/core_types/image_container.h>

#include <stdexcept>

namespace viame {

namespace {

namespace io = viame::image_ops;

// ----------------------------------------------------------------------------
io::structuring_element
element_for( std::string const& shape, double radius )
{
  if( shape == "disk" )  { return io::disk_element( radius ); }
  if( shape == "iline" ) { return io::line_i_element( radius ); }
  if( shape == "jline" ) { return io::line_j_element( radius ); }

  throw std::runtime_error( "unknown element shape '" + shape + "'" );
}

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class morphology::priv
{
public:
  priv( morphology& parent ) : m_parent( parent ) {}

  morphology& m_parent;
};

// ----------------------------------------------------------------------------
void
morphology
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.morphology" );
}

// ----------------------------------------------------------------------------
morphology
::~morphology()
{
}

// ----------------------------------------------------------------------------
void
morphology
::set_configuration_internal( kv::config_block_sptr )
{
}

// ----------------------------------------------------------------------------
bool
morphology
::check_configuration( kv::config_block_sptr in_config ) const
{
  auto config = this->get_configuration();
  config->merge_config( in_config );

  auto const radius = config->get_value< double >( "kernel_radius" );

  if( radius < 0 )
  {
    LOG_ERROR( logger(), "Config item kernel_radius should have been "
                         "non-negative but was " << radius );
  }

  return true;
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
morphology
::filter( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid input image." );
    return nullptr;
  }

  auto const& image = image_data->get_image();

  if( image.pixel_traits().type != kv::image_pixel_traits::BOOL )
  {
    LOG_ERROR( logger(), "Input format must be a bool" );
    return nullptr;
  }

  try
  {
    kv::image_of< bool > input( image );

    auto const element =
      element_for( get_element_shape(), get_kernel_radius() );

    auto const& operation = get_morphology();
    kv::image_of< bool > output;

    if( operation == "erode" )       { output = io::erode( input, element ); }
    else if( operation == "dilate" ) { output = io::dilate( input, element ); }
    else if( operation == "open" )   { output = io::opening( input, element ); }
    else if( operation == "close" )  { output = io::closing( input, element ); }
    else if( operation == "none" )   { output.copy_from( input ); }
    else
    {
      throw std::runtime_error( "unknown morphology '" + operation + "'" );
    }

    auto const& combination = get_channel_combination();

    if( combination == "union" )
    {
      output = io::combine_planes( output, true );
    }
    else if( combination == "intersection" )
    {
      output = io::combine_planes( output, false );
    }

    return std::make_shared< kv::simple_image_container >( output );
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( logger(), e.what() );
    return nullptr;
  }
}

} // end namespace viame
