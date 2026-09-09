/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_TEMPORAL_H
#define VIAME_IMAGE_OPS_TEMPORAL_H

#include "pixel.h"

#include <vital/types/image.h>

#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// How a running frame average weights the frames it has seen.
enum class average_mode
{
  window,       ///< Mean of the last N frames.
  cumulative,   ///< Mean of every frame since the last reset.
  exponential,  ///< Each frame weighted by a fixed factor against the past.
};

// ----------------------------------------------------------------------------
/// A running average of a stream of frames, kept in double precision.
///
/// The accumulator is double whatever the frame type is, and the result is
/// converted back on the way out, so a byte stream does not lose the
/// fractional part of its own average between frames.
///
/// It reproduces `arrows/vxl/average_frames` exactly, including one
/// behaviour that is not what the name suggests: once the window is full,
/// the frame subtracted from the running sum is the *newest* buffered frame
/// rather than the oldest, so a full window's average is not the mean of the
/// last N frames. Changing it would change the output of every motion
/// pipeline, so it is reproduced here and pinned by the recordings under
/// `tests/golden/vxl`. Fixing it is a behaviour change and needs its own
/// task and a re-recorded golden.
template < typename T >
class frame_averager
{
public:
  /// \param mode        Which weighting to use.
  /// \param window_size Frames in the window, for `window` mode.
  /// \param exp_weight  Weight of the new frame, for `exponential` mode.
  /// \param round       Round rather than truncate when converting the
  ///                    double accumulator back to an integral frame type.
  frame_averager( average_mode mode = average_mode::window,
                  size_t window_size = 10,
                  double exp_weight = 0.3,
                  bool round = false )
    : m_mode( mode ),
      m_window_size( window_size ),
      m_exp_weight( exp_weight ),
      m_round( round )
  {
    if( mode == average_mode::exponential &&
        ( exp_weight <= 0.0 || exp_weight >= 1.0 ) )
    {
      throw std::runtime_error( "invalid exponential averaging coefficient" );
    }
  }

  /// Forget every frame seen so far.
  void
  reset()
  {
    m_frame_count = 0;
    m_window.clear();
    m_average = kwiver::vital::image_of< double >();
  }

  /// Add \p input to the average and return the average after it.
  kwiver::vital::image_of< T >
  process( kwiver::vital::image_of< T > const& input )
  {
    if( resolution_changed( input ) )
    {
      reset();
    }

    switch( m_mode )
    {
      case average_mode::window:      update_window( input ); break;
      case average_mode::cumulative:  update_cumulative( input ); break;
      case average_mode::exponential: update_exponential( input ); break;
    }

    return convert_average( input );
  }

  /// As `process`, and also fill \p variance with the instantaneous estimate.
  ///
  /// The estimate is the product of the frame's distance from the average
  /// before the update and its distance from the average after, which is
  /// zero on the first frame of a given size. Averaging it over time
  /// approximates a per pixel variance.
  kwiver::vital::image_of< T >
  process( kwiver::vital::image_of< T > const& input,
           kwiver::vital::image_of< double >& variance )
  {
    auto const first = m_average.width() != input.width() ||
                       m_average.height() != input.height() ||
                       m_average.depth() != input.depth();

    if( first )
    {
      variance = kwiver::vital::image_of< double >(
        input.width(), input.height(), input.depth() );
      fill( variance, 0.0 );
      return process( input );
    }

    auto const before = absolute_difference( input, m_average );
    auto const average = process( input );
    auto const after = absolute_difference( input, average );

    variance = kwiver::vital::image_of< double >(
      input.width(), input.height(), input.depth() );

    for( size_t plane = 0; plane < input.depth(); ++plane )
    {
      for( size_t j = 0; j < input.height(); ++j )
      {
        for( size_t i = 0; i < input.width(); ++i )
        {
          variance( i, j, plane ) =
            before( i, j, plane ) * after( i, j, plane );
        }
      }
    }

    return average;
  }

  /// Frames currently in the window, for `window` mode.
  size_t frame_count() const { return m_window.size(); }

private:
  bool
  resolution_changed( kwiver::vital::image_of< T > const& input ) const
  {
    return input.width() != m_average.width() ||
           input.height() != m_average.height() ||
           input.depth() != m_average.depth();
  }

  static void
  fill( kwiver::vital::image_of< double >& image, double value )
  {
    for( size_t plane = 0; plane < image.depth(); ++plane )
    {
      for( size_t j = 0; j < image.height(); ++j )
      {
        for( size_t i = 0; i < image.width(); ++i )
        {
          image( i, j, plane ) = value;
        }
      }
    }
  }

  /// |a - b| in double, for any mix of frame and accumulator types.
  template < typename A, typename B >
  static kwiver::vital::image_of< double >
  absolute_difference( kwiver::vital::image_of< A > const& a,
                       kwiver::vital::image_of< B > const& b )
  {
    kwiver::vital::image_of< double > result( a.width(), a.height(),
                                              a.depth() );

    for( size_t plane = 0; plane < a.depth(); ++plane )
    {
      for( size_t j = 0; j < a.height(); ++j )
      {
        for( size_t i = 0; i < a.width(); ++i )
        {
          auto const left = static_cast< double >( a( i, j, plane ) );
          auto const right = static_cast< double >( b( i, j, plane ) );
          result( i, j, plane ) = ( left > right ) ? left - right
                                                   : right - left;
        }
      }
    }

    return result;
  }

  void
  seed( kwiver::vital::image_of< T > const& input )
  {
    m_average = kwiver::vital::image_of< double >(
      input.width(), input.height(), input.depth() );

    for( size_t plane = 0; plane < input.depth(); ++plane )
    {
      for( size_t j = 0; j < input.height(); ++j )
      {
        for( size_t i = 0; i < input.width(); ++i )
        {
          m_average( i, j, plane ) =
            static_cast< double >( input( i, j, plane ) );
        }
      }
    }
  }

  /// m_average = m_average * old_weight + input * new_weight
  void
  blend( kwiver::vital::image_of< T > const& input,
         double old_weight, double new_weight )
  {
    for( size_t plane = 0; plane < input.depth(); ++plane )
    {
      for( size_t j = 0; j < input.height(); ++j )
      {
        for( size_t i = 0; i < input.width(); ++i )
        {
          m_average( i, j, plane ) =
            old_weight * m_average( i, j, plane ) +
            new_weight * static_cast< double >( input( i, j, plane ) );
        }
      }
    }
  }

  void
  update_cumulative( kwiver::vital::image_of< T > const& input )
  {
    if( m_frame_count == 0 )
    {
      seed( input );
    }
    else
    {
      auto const weight = 1.0 / static_cast< double >( m_frame_count + 1 );
      blend( input, 1.0 - weight, weight );
    }

    ++m_frame_count;
  }

  void
  update_exponential( kwiver::vital::image_of< T > const& input )
  {
    if( m_frame_count == 0 )
    {
      seed( input );
    }
    else
    {
      blend( input, 1.0 - m_exp_weight, m_exp_weight );
    }

    ++m_frame_count;
  }

  void
  update_window( kwiver::vital::image_of< T > const& input )
  {
    auto const buffered = m_window.size();

    if( buffered == 0 )
    {
      seed( input );
    }
    else if( buffered < m_window_size )
    {
      auto const weight = 1.0 / ( static_cast< double >( buffered ) + 1.0 );
      blend( input, 1.0 - weight, weight );
    }
    else
    {
      // See the class comment: the frame removed is the newest buffered one,
      // not the oldest, and that is what the recordings pin
      auto const& removed = m_window[ buffered - 1 ];
      auto const weight = 1.0 / static_cast< double >( buffered );

      for( size_t plane = 0; plane < input.depth(); ++plane )
      {
        for( size_t j = 0; j < input.height(); ++j )
        {
          for( size_t i = 0; i < input.width(); ++i )
          {
            m_average( i, j, plane ) +=
              weight * ( static_cast< double >( input( i, j, plane ) ) -
                         static_cast< double >( removed( i, j, plane ) ) );
          }
        }
      }
    }

    kwiver::vital::image_of< T > copy;
    copy.copy_from( input );
    m_window.push_back( copy );

    if( m_window.size() > m_window_size )
    {
      m_window.pop_front();
    }
  }

  kwiver::vital::image_of< T >
  convert_average( kwiver::vital::image_of< T > const& like ) const
  {
    kwiver::vital::image_of< T > result( like.width(), like.height(),
                                         like.depth() );

    // Rounding only makes a difference going from the double accumulator to
    // an integral frame type
    constexpr bool integral = std::numeric_limits< T >::is_integer;
    bool const round = m_round && integral;

    for( size_t plane = 0; plane < like.depth(); ++plane )
    {
      for( size_t j = 0; j < like.height(); ++j )
      {
        for( size_t i = 0; i < like.width(); ++i )
        {
          auto const value = m_average( i, j, plane );
          result( i, j, plane ) = round ? round_pixel< T >( value )
                                        : static_cast< T >( value );
        }
      }
    }

    return result;
  }

  average_mode m_mode;
  size_t m_window_size;
  double m_exp_weight;
  bool m_round;

  size_t m_frame_count = 0;
  kwiver::vital::image_of< double > m_average;
  std::deque< kwiver::vital::image_of< T > > m_window;
};

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_TEMPORAL_H
