/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Template matching
///
/// What `cv::matchTemplate` did with `TM_CCOEFF_NORMED`, which is the only
/// method VIAME asks for: the stereo pairing walks a template along an
/// epipolar line and takes the best correlation.
///
/// Normalised cross-correlation of the mean-subtracted patches, so a change
/// in exposure between the two cameras does not move the peak -- which is
/// the whole reason that method rather than a plain difference.

#ifndef VIAME_IMAGE_OPS_MATCH_H
#define VIAME_IMAGE_OPS_MATCH_H

#include <viame/core_types/image.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// The correlation surface of \p pattern over \p image, `TM_CCOEFF_NORMED`.
///
/// The result is `image.width() - pattern.width() + 1` by the same in the
/// other axis: one score per position the pattern fits entirely, which is
/// what `cv::matchTemplate` returns. Scores run -1 to 1 and are float,
/// because a correlation is not a pixel value and rounding it to one throws
/// the answer away.
///
/// Where a patch has no variance -- a flat template, or a flat window under
/// it -- the correlation is undefined; OpenCV returns zero there when only
/// one of the two is flat and one when both are, and so does this.
///
/// All planes are used together, as OpenCV does for a multi-channel input --
/// one score, not one per channel. The mean subtracted is per plane, which
/// is also OpenCV's and is not the same as one mean over the whole patch.
template < typename T >
kwiver::vital::image_of< float >
match_template_ncc( kwiver::vital::image_of< T > const& image,
                    kwiver::vital::image_of< T > const& pattern )
{
  if( image.depth() != pattern.depth() )
  {
    throw std::invalid_argument(
      "match_template_ncc: the two differ in plane count" );
  }

  if( pattern.width() == 0 || pattern.height() == 0 )
  {
    throw std::invalid_argument( "match_template_ncc: the pattern is empty" );
  }

  if( pattern.width() > image.width() || pattern.height() > image.height() )
  {
    throw std::invalid_argument(
      "match_template_ncc: the pattern is larger than the image" );
  }

  auto const out_w = image.width() - pattern.width() + 1;
  auto const out_h = image.height() - pattern.height() + 1;

  auto const planes = pattern.depth();
  auto const per_plane = static_cast< double >(
    pattern.width() * pattern.height() );

  // The mean is taken **per plane**, not over the patch as a whole. That is
  // what OpenCV does, and on a colour image the two differ by up to a whole
  // unit of correlation -- a colour patch has a different average in each
  // channel, and subtracting one joint average leaves that offset in as
  // signal. On a single plane image the two coincide, which is why only the
  // colour case says so.
  std::vector< double > pattern_mean( planes, 0.0 );

  for( size_t plane = 0; plane < planes; ++plane )
  {
    double total = 0.0;

    for( size_t j = 0; j < pattern.height(); ++j )
    {
      for( size_t i = 0; i < pattern.width(); ++i )
      {
        total += static_cast< double >( pattern( i, j, plane ) );
      }
    }

    pattern_mean[ plane ] = total / per_plane;
  }

  double pattern_energy = 0.0;

  for( size_t plane = 0; plane < planes; ++plane )
  {
    for( size_t j = 0; j < pattern.height(); ++j )
    {
      for( size_t i = 0; i < pattern.width(); ++i )
      {
        auto const d = static_cast< double >( pattern( i, j, plane ) ) -
                       pattern_mean[ plane ];
        pattern_energy += d * d;
      }
    }
  }

  kwiver::vital::image_of< float > out( out_w, out_h, 1 );

  std::vector< double > window_mean( planes, 0.0 );

  for( size_t top = 0; top < out_h; ++top )
  {
    for( size_t left = 0; left < out_w; ++left )
    {
      for( size_t plane = 0; plane < planes; ++plane )
      {
        double total = 0.0;

        for( size_t j = 0; j < pattern.height(); ++j )
        {
          for( size_t i = 0; i < pattern.width(); ++i )
          {
            total +=
              static_cast< double >( image( left + i, top + j, plane ) );
          }
        }

        window_mean[ plane ] = total / per_plane;
      }

      double cross = 0.0;
      double window_energy = 0.0;

      for( size_t plane = 0; plane < planes; ++plane )
      {
        for( size_t j = 0; j < pattern.height(); ++j )
        {
          for( size_t i = 0; i < pattern.width(); ++i )
          {
            auto const a =
              static_cast< double >( image( left + i, top + j, plane ) ) -
              window_mean[ plane ];
            auto const b =
              static_cast< double >( pattern( i, j, plane ) ) -
              pattern_mean[ plane ];

            cross += a * b;
            window_energy += a * a;
          }
        }
      }

      auto const denominator = std::sqrt( window_energy * pattern_energy );

      // Both flat is a perfect match by convention; one flat is undefined
      // and answers zero. OpenCV does the same, and the two cases are worth
      // separating: a flat template against flat water should score, and a
      // flat template against texture should not.
      double score = 0.0;

      if( denominator > 0.0 )
      {
        score = cross / denominator;
      }
      else if( window_energy == 0.0 && pattern_energy == 0.0 )
      {
        score = 1.0;
      }

      out( left, top, 0 ) = static_cast< float >( score );
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
