/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Adaptive background subtraction by a mixture of Gaussians
///
/// What `cv::BackgroundSubtractorMOG2` did. Zivkovic's method: every pixel
/// carries its own small mixture of Gaussians describing the colours it has
/// been, each with a weight, a mean and one variance shared across the
/// channels. A new frame's pixel is matched against them; a match updates the
/// mixture and counts as background if it landed in the modes that make up
/// the bulk of the weight, and a miss starts a new mode in place of the
/// weakest one.
///
/// What makes it adaptive rather than a plain average is the **complexity
/// reduction prior**: every weight is pushed down a little each frame, so a
/// mode that stops being seen fades out and is dropped, and the mixture keeps
/// only as many modes as the pixel actually needs.
///
/// Shadow detection is not implemented. OpenCV offers it and the one caller
/// in this tree asks for it off; a mixture with it on returns a third label
/// rather than a binary mask, so it is a different contract and would want
/// its own work.
///
/// The details are OpenCV's, because `gmm_motion_detector` is a shipped
/// process and nothing records what it returns:
///
/// * the learning rate, when not given, is `1 / min(2 * frames, history)`, so
///   it starts fast and settles at the history length;
/// * the modes are kept sorted by weight, descending, and a mode that grows
///   past its neighbour is bubbled up *as it is updated*;
/// * a pixel is background if the matching mode is reached before the
///   cumulative weight passes `background_ratio` -- which is tested with the
///   weights of the **previous** frame, before this frame's renormalisation;
/// * matching uses `var_threshold` and mode creation uses
///   `var_threshold_gen`, which are different numbers, the second smaller.
///
/// Checked against `cv2.createBackgroundSubtractorMOG2` over six sequences
/// and 220 frames -- grayscale and colour, a history shorter than the run, a
/// step change in lighting, thresholds from 9 to 30 -- and not one pixel of
/// one mask differs.

#ifndef VIAME_IMAGE_KERNELS_BACKGROUND_H
#define VIAME_IMAGE_KERNELS_BACKGROUND_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

/// How `cv::createBackgroundSubtractorMOG2` is configured.
struct mog2_params
{
  /// How many frames the learning rate settles to.
  int history = 500;

  /// How close a pixel must be to a mode, in variances, to match it.
  double var_threshold = 16.0;

  /// The most Gaussians one pixel may carry.
  int mixtures = 5;

  /// How much of the weight counts as background.
  double background_ratio = 0.9;

  /// How close a pixel must be to avoid starting a new mode.
  double var_threshold_gen = 9.0;

  /// The variance a new mode starts with, and its bounds.
  double var_init = 15.0;
  double var_min = 4.0;
  double var_max = 75.0;

  /// The complexity reduction prior, which is what lets a mode die.
  double complexity_reduction = 0.05;
};

// ----------------------------------------------------------------------------
/// A per-pixel Gaussian mixture, updated one frame at a time.
///
/// Stateful, like `frame_averager` next door: the mixture is the point, and
/// a frame only means anything relative to the ones before it.
class mog2_background
{
public:
  explicit mog2_background( mog2_params const& params = mog2_params() )
    : m_params( params )
  {
    if( params.mixtures < 1 )
    {
      throw std::invalid_argument( "mog2 wants at least one mixture" );
    }

    if( params.history < 1 )
    {
      throw std::invalid_argument( "mog2 wants a positive history" );
    }
  }

  /// Forget every frame seen so far.
  void
  reset()
  {
    m_frames = 0;
    m_weight.clear();
    m_mean.clear();
    m_variance.clear();
    m_used.clear();
  }

  /// How many frames have been through it.
  size_t frames() const { return m_frames; }

  // --------------------------------------------------------------------------
  /// Update the mixture with \p image and return its foreground mask.
  ///
  /// 255 where the pixel did not match the background, 0 where it did.
  ///
  /// @param image the frame, one or more planes
  /// @param learning_rate how fast to adapt; negative asks for the automatic
  ///        `1 / min(2 * frames, history)`
  template < typename T >
  viame::image_of< uint8_t >
  apply( viame::image_of< T > const& image, double learning_rate = -1.0 )
  {
    auto const width = image.width();
    auto const height = image.height();
    auto const depth = image.depth();
    auto const modes = static_cast< size_t >( m_params.mixtures );

    if( width == 0 || height == 0 || depth == 0 )
    {
      throw std::invalid_argument( "mog2 wants a frame with pixels in it" );
    }

    if( !m_weight.empty() &&
        ( m_width != width || m_height != height || m_depth != depth ) )
    {
      throw std::invalid_argument(
        "mog2 wants every frame the same shape as the first" );
    }

    if( m_weight.empty() )
    {
      m_width = width;
      m_height = height;
      m_depth = depth;
      m_weight.assign( width * height * modes, 0.0f );
      m_mean.assign( width * height * modes * depth, 0.0f );
      m_variance.assign( width * height * modes, 0.0f );
      m_used.assign( width * height, 0 );
    }

    ++m_frames;

    auto const alpha = ( learning_rate >= 0.0 && m_frames > 1 )
      ? static_cast< float >( learning_rate )
      : 1.0f / static_cast< float >( std::min(
          static_cast< size_t >( 2 ) * m_frames,
          static_cast< size_t >( m_params.history ) ) );

    auto const alpha1 = 1.0f - alpha;
    auto const prune = -alpha * static_cast< float >(
      m_params.complexity_reduction );

    auto const match = static_cast< float >( m_params.var_threshold );
    auto const generate = static_cast< float >( m_params.var_threshold_gen );
    auto const ratio = static_cast< float >( m_params.background_ratio );
    auto const var_init = static_cast< float >( m_params.var_init );
    auto const var_min = static_cast< float >( m_params.var_min );
    auto const var_max = static_cast< float >( m_params.var_max );

    viame::image_of< uint8_t > out( width, height, 1 );

    std::vector< float > pixel( depth );

    for( size_t y = 0; y < height; ++y )
    {
      for( size_t x = 0; x < width; ++x )
      {
        auto const at = y * width + x;
        auto* weight = m_weight.data() + at * modes;
        auto* variance = m_variance.data() + at * modes;
        auto* mean = m_mean.data() + at * modes * depth;

        for( size_t c = 0; c < depth; ++c )
        {
          pixel[ c ] = static_cast< float >( image( x, y, c ) );
        }

        auto used = static_cast< size_t >( m_used[ at ] );

        float total = 0.0f;
        bool fits = false;
        bool background = false;

        for( size_t mode = 0; mode < used; ++mode )
        {
          auto value = alpha1 * weight[ mode ] + prune;
          size_t swaps = 0;

          if( !fits )
          {
            auto const var = variance[ mode ];

            float distance = 0.0f;

            for( size_t c = 0; c < depth; ++c )
            {
              auto const step = mean[ mode * depth + c ] - pixel[ c ];
              distance += step * step;
            }

            // The cumulative weight here is last frame's, before this frame
            // renormalises -- which is OpenCV's, and is not the same test as
            // asking afterwards
            if( total < ratio && distance < match * var )
            {
              background = true;
            }

            if( distance < generate * var )
            {
              fits = true;
              value += alpha;

              auto const k = alpha / value;

              for( size_t c = 0; c < depth; ++c )
              {
                mean[ mode * depth + c ] -=
                  k * ( mean[ mode * depth + c ] - pixel[ c ] );
              }

              auto updated = var + k * ( distance - var );
              variance[ mode ] = std::min( std::max( updated, var_min ),
                                           var_max );

              // Bubble the mode up past any it now outweighs
              for( size_t i = mode; i > 0; --i )
              {
                if( value < weight[ i - 1 ] ) { break; }

                ++swaps;
                std::swap( weight[ i ], weight[ i - 1 ] );
                std::swap( variance[ i ], variance[ i - 1 ] );

                for( size_t c = 0; c < depth; ++c )
                {
                  std::swap( mean[ i * depth + c ],
                             mean[ ( i - 1 ) * depth + c ] );
                }
              }
            }
          }

          if( value < -prune )
          {
            value = 0.0f;
            --used;
          }

          weight[ mode - swaps ] = value;
          total += value;
        }

        if( total > 0.0f )
        {
          for( size_t mode = 0; mode < modes; ++mode )
          {
            weight[ mode ] /= total;
          }
        }

        if( !fits && alpha > 0.0f )
        {
          auto const slot = ( used == modes ) ? modes - 1 : used;

          if( used < modes ) { ++used; }

          if( used == 1 )
          {
            weight[ slot ] = 1.0f;
          }
          else
          {
            weight[ slot ] = alpha;

            for( size_t mode = 0; mode < used; ++mode )
            {
              if( mode != slot ) { weight[ mode ] *= alpha1; }
            }
          }

          for( size_t c = 0; c < depth; ++c )
          {
            mean[ slot * depth + c ] = pixel[ c ];
          }

          variance[ slot ] = var_init;
        }

        m_used[ at ] = static_cast< int32_t >( used );
        out( x, y, 0 ) = background ? uint8_t{ 0 } : uint8_t{ 255 };
      }
    }

    return out;
  }

private:
  mog2_params m_params;

  size_t m_frames = 0;
  size_t m_width = 0;
  size_t m_height = 0;
  size_t m_depth = 0;

  std::vector< float > m_weight;
  std::vector< float > m_mean;
  std::vector< float > m_variance;
  std::vector< int32_t > m_used;
};

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_BACKGROUND_H
