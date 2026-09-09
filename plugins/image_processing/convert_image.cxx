/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "convert_image.h"

#include <channels.h>
#include <convert.h>
#include <dispatch.h>

#include <vital/types/image_container.h>

#include <cstdint>
#include <random>
#include <string>

namespace viame {

namespace {

namespace io = viame::image_ops;

// ----------------------------------------------------------------------------
/// Number of samples the percentile stretch takes.
///
/// Larger than any image anyone runs this on, so the stretch is effectively
/// over every pixel; the sampler caps it at the image size. The value is what
/// arrows/vxl passed, and the sampled set is what fixes the percentile, so it
/// is not free to change.
constexpr size_t percentile_sampling_points = 100000000;

// ----------------------------------------------------------------------------
/// Convert \p input to \p Out, then apply the output shaping.
template < typename Out, typename In >
kwiver::vital::image
convert_to( kwiver::vital::image_of< In > const& input,
            double scale_factor, double percentile_norm,
            bool force_three_channel )
{
  kwiver::vital::image_of< Out > output;

  if( percentile_norm >= 0.0 )
  {
    output = io::percentile_stretch< Out >(
      input, percentile_norm, 1.0 - percentile_norm,
      percentile_sampling_points );
  }
  else if( scale_factor == 0.0 || scale_factor == 1.0 )
  {
    output = io::cast< Out >( input );
  }
  else
  {
    output = io::scale< Out >( input, scale_factor );
  }

  if( force_three_channel )
  {
    output = io::force_three_channels( output );
  }

  return output;
}

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class convert_image::priv
{
public:
  priv( convert_image& parent ) : m_parent( parent ) {}

  convert_image& m_parent;

  /// Reduce to one plane, or grey the whole image with some probability.
  ///
  /// Only one of the two happens: asking for a single channel wins, which is
  /// what arrows/vxl does and what the pipelines that set both expect.
  template < typename T >
  kwiver::vital::image_of< T >
  apply_transforms( kwiver::vital::image_of< T > const& input )
  {
    if( m_parent.get_single_channel() && input.depth() != 1 )
    {
      return io::combine_channels( input );
    }

    auto const fraction = m_parent.get_random_grayscale();

    if( fraction > 0.0 && m_distribution( m_engine ) < fraction )
    {
      auto const grey = io::combine_channels( input );
      return io::broadcast_plane( grey, 0, input.depth() );
    }

    return input;
  }

  /// Dispatch on the configured output format.
  template < typename In >
  kwiver::vital::image
  convert( kwiver::vital::image_of< In > const& input )
  {
    auto const& format = m_parent.get_format();
    auto const scale_factor = m_parent.get_scale_factor();
    auto const percentile_norm = m_parent.get_percentile_norm();
    auto const force_three = m_parent.get_force_three_channel();

#define VIAME_CONVERT_CASE( name, type )                                \
    if( format == name )                                                \
    {                                                                   \
      return convert_to< type >( input, scale_factor, percentile_norm,  \
                                 force_three );                         \
    }

    VIAME_CONVERT_CASE( "copy", In )
    VIAME_CONVERT_CASE( "byte", uint8_t )
    VIAME_CONVERT_CASE( "sbyte", int8_t )
    VIAME_CONVERT_CASE( "uint16", uint16_t )
    VIAME_CONVERT_CASE( "int16", int16_t )
    VIAME_CONVERT_CASE( "uint32", uint32_t )
    VIAME_CONVERT_CASE( "int32", int32_t )
    VIAME_CONVERT_CASE( "uint64", uint64_t )
    VIAME_CONVERT_CASE( "int64", int64_t )
    VIAME_CONVERT_CASE( "float", float )
    VIAME_CONVERT_CASE( "double", double )

#undef VIAME_CONVERT_CASE

    throw std::runtime_error( "invalid output format '" + format + "'" );
  }

  // The augmentation is deliberately unseeded, as in arrows/vxl: it is a
  // training augmentation, and reproducing a particular draw is not the point
  std::random_device m_random_device;
  std::mt19937 m_engine{ m_random_device() };
  std::uniform_real_distribution< double > m_distribution{ 0.0, 1.0 };
};

// ----------------------------------------------------------------------------
void
convert_image
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.image_processing.convert_image" );
}

// ----------------------------------------------------------------------------
convert_image
::~convert_image()
{
}

// ----------------------------------------------------------------------------
void
convert_image
::set_configuration_internal( kv::config_block_sptr )
{
}

// ----------------------------------------------------------------------------
bool
convert_image
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
convert_image
::filter( kv::image_container_sptr image_data )
{
  if( !image_data )
  {
    return nullptr;
  }

  if( get_format() == "disable" )
  {
    return image_data;
  }

  try
  {
    auto const result = io::dispatch_pixel_type(
      image_data->get_image(),
      [ this ]( auto const& typed ) -> kv::image
      {
        return d->convert( d->apply_transforms( typed ) );
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
