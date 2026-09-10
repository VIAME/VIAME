/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "color_commonality.h"

#include <image_ops/commonality.h>
#include <image_ops/dispatch.h>

#include <viame/core_types/image_container.h>

#include <stdexcept>

namespace viame {

namespace {

namespace io = viame::image_ops;

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class color_commonality::priv
{
public:
  priv( color_commonality& parent ) : m_parent( parent ) {}

  color_commonality& m_parent;
};

// ----------------------------------------------------------------------------
void
color_commonality
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.color_commonality" );
}

// ----------------------------------------------------------------------------
color_commonality
::~color_commonality()
{
}

// ----------------------------------------------------------------------------
void
color_commonality
::set_configuration_internal( kv::config_block_sptr )
{
}

// ----------------------------------------------------------------------------
bool
color_commonality
::check_configuration( kv::config_block_sptr in_config ) const
{
  auto config = this->get_configuration();
  config->merge_config( in_config );

  auto const color = config->get_value< unsigned >(
    "color_resolution_per_channel" );
  auto const intensity = config->get_value< unsigned >(
    "intensity_resolution" );

  if( !io::is_power_of_two( color ) )
  {
    LOG_ERROR( logger(), "color_resolution_per_channel must be a power of 2, "
                         "but instead is: " << color );
    return false;
  }

  if( !io::is_power_of_two( intensity ) )
  {
    LOG_ERROR( logger(), "intensity_resolution must be a power of 2, "
                         "but instead is: " << intensity );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
/// Replace each pixel with how common its colour is.
///
/// One divergence from `vxl_color_commonality`: in grid mode that
/// implementation builds each tile's region with the corner coordinates in
/// the wrong order, so most tiles come out empty and most of the output is
/// never written, giving whatever the heap held. Here grid mode computes what
/// the option describes, a per tile commonality. No shipped pipeline sets
/// grid_image.
kv::image_container_sptr
color_commonality
::filter( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid input image." );
    return nullptr;
  }

  auto const scale = get_output_scale();
  bool const grid = get_grid_image();
  auto const columns = get_grid_resolution_width();
  auto const rows = get_grid_resolution_height();
  auto const color_bins = get_color_resolution_per_channel();
  auto const intensity_bins = get_intensity_resolution();

  try
  {
    auto const result = io::dispatch_pixel_type(
      image_data->get_image(),
      [ & ]( auto const& typed ) -> kv::image
      {
        using pixel_t = io::pixel_type_t< decltype( typed ) >;

        if constexpr( !std::numeric_limits< pixel_t >::is_integer )
        {
          throw std::runtime_error( "input must be an integer type" );
        }
        else
        {
          auto const bins =
            ( typed.depth() == 1 ) ? intensity_bins : color_bins;

          return grid
            ? io::color_commonality_grid( typed, bins, scale, columns, rows )
            : io::color_commonality( typed, bins, scale );
        }
      } );

    return std::make_shared< kv::simple_image_container >( result );
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( logger(), e.what() );
    return nullptr;
  }
}

} // end namespace viame
