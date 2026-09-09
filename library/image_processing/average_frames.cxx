/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "average_frames.h"

#include <image_ops/dispatch.h>
#include <image_ops/temporal.h>

#include <vital/types/image_container.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <typeindex>

namespace viame {

namespace {

namespace io = viame::image_ops;

// ----------------------------------------------------------------------------
io::average_mode
mode_from_string( std::string const& name )
{
  if( name == "window" ) { return io::average_mode::window; }
  if( name == "cumulative" ) { return io::average_mode::cumulative; }
  if( name == "exponential" ) { return io::average_mode::exponential; }

  throw std::runtime_error( "unknown averaging mode '" + name + "'" );
}

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class average_frames::priv
{
public:
  priv( average_frames& parent ) : m_parent( parent ) {}

  average_frames& m_parent;

  /// The averager for one pixel type, made on first sight of that type.
  ///
  /// A pipeline feeds one type, but the algorithm is not told which until a
  /// frame arrives, and the running average has to survive between frames.
  template < typename T >
  io::frame_averager< T >&
  averager()
  {
    auto const key = std::type_index( typeid( T ) );
    auto found = m_averagers.find( key );

    if( found == m_averagers.end() )
    {
      auto created = std::make_shared< io::frame_averager< T > >(
        mode_from_string( m_parent.get_type() ),
        m_parent.get_window_size(),
        m_parent.get_exp_weight(),
        m_parent.get_round() );

      found = m_averagers.emplace( key, std::move( created ) ).first;
    }

    return *std::static_pointer_cast< io::frame_averager< T > >(
      found->second );
  }

  void reset() { m_averagers.clear(); }

private:
  std::map< std::type_index, std::shared_ptr< void > > m_averagers;
};

// ----------------------------------------------------------------------------
void
average_frames
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.average_frames" );
}

// ----------------------------------------------------------------------------
average_frames
::~average_frames()
{
}

// ----------------------------------------------------------------------------
void
average_frames
::set_configuration_internal( kv::config_block_sptr )
{
  // The averagers hold the old settings and a partly accumulated average
  d->reset();
}

// ----------------------------------------------------------------------------
bool
average_frames
::check_configuration( kv::config_block_sptr in_config ) const
{
  auto config = this->get_configuration();
  config->merge_config( in_config );

  auto const type = config->get_value< std::string >( "type" );

  if( type != "window" && type != "cumulative" && type != "exponential" )
  {
    LOG_ERROR( logger(), "Unknown averaging type '" << type << "'" );
    return false;
  }

  if( type == "exponential" )
  {
    auto const weight = config->get_value< double >( "exp_weight" );

    if( weight <= 0.0 || weight > 1.0 )
    {
      LOG_ERROR( logger(), "exp_weight must be in (0, 1] but was " << weight );
      return false;
    }
  }

  if( type == "window" && config->get_value< unsigned >( "window_size" ) < 1 )
  {
    LOG_ERROR( logger(), "window_size must be at least 1" );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
average_frames
::filter( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid input image." );
    return image_data;
  }

  bool const want_variance = get_output_variance();

  try
  {
    auto const result = io::dispatch_pixel_type(
      image_data->get_image(),
      [ this, want_variance ]( auto const& typed ) -> kv::image
      {
        using pixel_t = io::pixel_type_t< decltype( typed ) >;

        auto& averager = d->averager< pixel_t >();

        if( !want_variance )
        {
          return averager.process( typed );
        }

        kv::image_of< double > variance;
        averager.process( typed, variance );
        return variance;
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
