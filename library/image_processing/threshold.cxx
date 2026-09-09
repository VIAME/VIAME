/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "threshold.h"

#include <image_ops/dispatch.h>
#include <image_ops/threshold.h>

#include <vital/types/image_container.h>

#include <stdexcept>

namespace viame {

namespace {

namespace io = viame::image_ops;

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class threshold::priv
{
public:
  priv( threshold& parent ) : m_parent( parent ) {}

  threshold& m_parent;
};

// ----------------------------------------------------------------------------
void
threshold
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.threshold" );
}

// ----------------------------------------------------------------------------
threshold
::~threshold()
{
}

// ----------------------------------------------------------------------------
void
threshold
::set_configuration_internal( kv::config_block_sptr )
{
}

// ----------------------------------------------------------------------------
bool
threshold
::check_configuration( kv::config_block_sptr in_config ) const
{
  auto config = this->get_configuration();
  config->merge_config( in_config );

  auto const type = config->get_value< std::string >( "type" );
  auto const value = config->get_value< double >( "threshold" );

  if( type != "absolute" && type != "percentile" )
  {
    LOG_ERROR( logger(), "Unknown threshold type '" << type << "'" );
    return false;
  }

  if( type == "percentile" && ( value < 0.0 || value > 1.0 ) )
  {
    LOG_ERROR( logger(), "threshold must be in [0, 1] but instead was "
                         << value );
  }

  return true;
}

// ----------------------------------------------------------------------------
/// Threshold the image, above the given value or above a percentile of it.
///
/// One divergence from `vxl_threshold`: in percentile mode on an image with
/// more than one plane, that implementation sized its output to a single
/// plane and then handed a view of it to a routine that resizes what it is
/// given, which detached the view and left the output never written. The
/// result was whatever the heap held. Here every plane is thresholded against
/// the percentile and the output keeps the input's plane count, the same rule
/// absolute mode follows. No shipped pipeline uses that path: the one
/// pipeline that thresholds does so in absolute mode on a single plane.
kv::image_container_sptr
threshold
::filter( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid input image." );
    return nullptr;
  }

  auto const value = get_threshold();
  bool const percentile = ( get_type() == "percentile" );

  try
  {
    auto const result = io::dispatch_pixel_type(
      image_data->get_image(),
      [ value, percentile ]( auto const& typed ) -> kv::image
      {
        using pixel_t = io::pixel_type_t< decltype( typed ) >;

        if( percentile )
        {
          return io::threshold_percentile( typed, value );
        }

        return io::threshold_above( typed, static_cast< pixel_t >( value ) );
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
