/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "core_image_io.h"

#include <image_ops/convert.h>
#include <image_ops/dispatch.h>
#include <image_ops/stretch.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/core_types/image_container.h>
#include <viame/algorithm_framework/util/tokenize.h>

#include <kwiversys/SystemTools.hxx>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace viame {

namespace {

namespace io = viame::image_ops;
typedef kwiversys::SystemTools ST;

// ----------------------------------------------------------------------------
/// Which image_io actually touches the file.
///
/// Decoding stays borrowed until the codecs come in-house; everything this
/// class is for happens to the pixels afterwards.
constexpr char const* decoder_name = "ocv";

// ----------------------------------------------------------------------------
/// `<dir>/<stem>_<index><ext>`, or the plain name for index 0.
std::string
plane_filename( std::string const& filename, unsigned index )
{
  auto const directory = ST::GetParentDirectory( filename );
  auto const name = ST::GetFilenameName( filename );
  auto const stem = ST::GetFilenameWithoutLastExtension( name );
  auto const extension = ST::GetFilenameLastExtension( name );

  auto const suffix = ( index > 0 ) ? "_" + std::to_string( index )
                                    : std::string();

  auto const leaf = stem + suffix + extension;

  return directory.empty() ? leaf : directory + "/" + leaf;
}

// ----------------------------------------------------------------------------
/// Stack \p planes, which all have to share a size, into one image.
template < typename T >
kwiver::vital::image_of< T >
stack_planes( std::vector< kwiver::vital::image_of< T > > const& planes )
{
  size_t depth = 0;

  for( auto const& plane : planes )
  {
    depth += plane.depth();
  }

  kwiver::vital::image_of< T > result(
    planes[ 0 ].width(), planes[ 0 ].height(), depth );

  size_t target = 0;

  for( auto const& plane : planes )
  {
    for( size_t source = 0; source < plane.depth(); ++source, ++target )
    {
      for( size_t j = 0; j < plane.height(); ++j )
      {
        for( size_t i = 0; i < plane.width(); ++i )
        {
          result( i, j, target ) = plane( i, j, source );
        }
      }
    }
  }

  return result;
}

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class core_image_io::priv
{
public:
  priv( core_image_io& parent ) : m_parent( parent ) {}

  core_image_io& m_parent;

  /// The borrowed decoder, made once and kept.
  kwiver::vital::algo::image_io_sptr
  decoder() const
  {
    if( !m_decoder )
    {
      kwiver::vital::implementation_factory_by_name<
        kwiver::vital::algo::image_io > factory;
      m_decoder = factory.create(
        decoder_name, kwiver::vital::config_block::empty_config() );

      if( !m_decoder )
      {
        throw std::runtime_error(
          std::string( "image_io '" ) + decoder_name + "' is not registered" );
      }
    }

    return m_decoder;
  }

  /// The two numbers of intensity_range, as the pixel type.
  template < typename T >
  void
  manual_bounds( T& low, T& high ) const
  {
    std::vector< std::string > tokens;
    kwiver::vital::tokenize( m_parent.get_intensity_range(), tokens, " ",
                             true );

    double values[ 2 ] = { 0.0, 255.0 };

    for( size_t index = 0; index < 2 && index < tokens.size(); ++index )
    {
      values[ index ] = std::stod( tokens[ index ] );
    }

    low = static_cast< T >( values[ 0 ] );
    high = static_cast< T >( values[ 1 ] );
  }

  /// Apply the configured range handling, producing \p Out pixels.
  template < typename Out, typename In >
  kwiver::vital::image_of< Out >
  convert( kwiver::vital::image_of< In > const& input ) const
  {
    // A byte target scales straight into the byte rather than through a
    // double image; the two differ in where the top value lands
    if constexpr( std::is_same< Out, uint8_t >::value )
    {
      if( m_parent.get_auto_stretch() )
      {
        return io::stretch_to_byte( input );
      }

      if( m_parent.get_manual_stretch() )
      {
        In low;
        In high;
        manual_bounds( low, high );
        return io::cast< Out >(
          io::stretch_range_limited( input, low, high, 0.0, 255.0 ) );
      }

      return io::cast< Out >( input );
    }
    else
    {
      double low;
      double high;
      io::stretch_target< Out >( low, high );

      if( m_parent.get_auto_stretch() )
      {
        return io::cast< Out >( io::stretch_range( input, low, high ) );
      }

      if( m_parent.get_manual_stretch() )
      {
        In source_low;
        In source_high;
        manual_bounds( source_low, source_high );
        return io::cast< Out >(
          io::stretch_range_limited( input, source_low, source_high,
                                     low, high ) );
      }

      return io::cast< Out >( input );
    }
  }

private:
  mutable kwiver::vital::algo::image_io_sptr m_decoder;
};

// ----------------------------------------------------------------------------
void
core_image_io
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.core_image_io" );
}

// ----------------------------------------------------------------------------
core_image_io
::~core_image_io()
{
}

// ----------------------------------------------------------------------------
void
core_image_io
::set_configuration_internal( kv::config_block_sptr )
{
}

// ----------------------------------------------------------------------------
bool
core_image_io
::check_configuration( kv::config_block_sptr in_config ) const
{
  auto config = this->get_configuration();
  config->merge_config( in_config );

  if( config->get_value< bool >( "auto_stretch" ) &&
      config->get_value< bool >( "manual_stretch" ) )
  {
    LOG_ERROR( logger(), "auto_stretch and manual_stretch are exclusive" );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
core_image_io
::load_( std::string const& filename ) const
{
  auto const loaded = d->decoder()->load( filename );

  if( !loaded )
  {
    return loaded;
  }

  auto image = loaded->get_image();

  // Sibling files carry the remaining channels when they were written apart
  if( get_split_channels() )
  {
    image = io::dispatch_pixel_type(
      image,
      [ & ]( auto const& typed ) -> kv::image
      {
        using pixel_t = io::pixel_type_t< decltype( typed ) >;

        std::vector< kv::image_of< pixel_t > > planes{ typed };

        for( unsigned index = 1;; ++index )
        {
          auto const plane_file = plane_filename( filename, index );

          if( !ST::FileExists( plane_file ) )
          {
            break;
          }

          auto const plane = d->decoder()->load( plane_file );
          kv::image_of< pixel_t > typed_plane( plane->get_image() );

          if( typed_plane.width() != typed.width() ||
              typed_plane.height() != typed.height() )
          {
            throw std::runtime_error( "input channel size difference" );
          }

          planes.push_back( typed_plane );
        }

        return planes.size() == 1 ? kv::image( typed )
                                  : kv::image( stack_planes( planes ) );
      } );
  }

  auto const converted = io::dispatch_pixel_type(
    image,
    [ & ]( auto const& typed ) -> kv::image
    {
      using pixel_t = io::pixel_type_t< decltype( typed ) >;

      return get_force_byte()
        ? kv::image( d->convert< uint8_t >( typed ) )
        : kv::image( d->convert< pixel_t >( typed ) );
    } );

  auto result = std::make_shared< kv::simple_image_container >( converted );
  result->set_metadata( loaded->get_metadata() );
  return result;
}

// ----------------------------------------------------------------------------
void
core_image_io
::save_( std::string const& filename, kv::image_container_sptr data ) const
{
  if( !data )
  {
    return;
  }

  auto const converted = io::dispatch_pixel_type(
    data->get_image(),
    [ & ]( auto const& typed ) -> kv::image
    {
      using pixel_t = io::pixel_type_t< decltype( typed ) >;

      return get_force_byte()
        ? kv::image( d->convert< uint8_t >( typed ) )
        : kv::image( d->convert< pixel_t >( typed ) );
    } );

  if( !get_split_channels() || converted.depth() == 1 )
  {
    d->decoder()->save(
      filename,
      std::make_shared< kv::simple_image_container >( converted ) );
    return;
  }

  io::dispatch_pixel_type(
    converted,
    [ & ]( auto const& typed ) -> int
    {
      using pixel_t = io::pixel_type_t< decltype( typed ) >;

      for( size_t plane = 0; plane < typed.depth(); ++plane )
      {
        kv::image_of< pixel_t > single( typed.width(), typed.height(), 1 );

        for( size_t j = 0; j < typed.height(); ++j )
        {
          for( size_t i = 0; i < typed.width(); ++i )
          {
            single( i, j, 0 ) = typed( i, j, plane );
          }
        }

        d->decoder()->save(
          plane_filename( filename, static_cast< unsigned >( plane ) ),
          std::make_shared< kv::simple_image_container >( single ) );
      }

      return 0;
    } );
}

} // end namespace viame
