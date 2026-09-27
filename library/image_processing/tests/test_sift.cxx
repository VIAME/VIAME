/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The SIFT port against what OpenCV computes.
///
/// `tests/golden/sift/opencv.json` is recorded by
/// `tests/golden/sift/record_from_opencv.py` under the install's own cv2 --
/// SIFT is in every wheel, so unlike SURF no special build is needed. The
/// tiles are the grayscale PNGs `tests/golden/surf/` already carries, so the
/// colour conversion is not a variable.
///
/// This is a port rather than a reimplementation, so the bar is agreement:
/// the same keypoints, in the same places, with descriptors pointing the same
/// way. It is **not** bit exactness, and the reason is in sift.cxx's header --
/// `cv::hal::exp32f` is a vectorised polynomial this does not reproduce, and
/// the float bilinear resize that doubles the base image is within one ULP of
/// OpenCV's rather than equal to it. Both move a keypoint that sits on a
/// threshold, so the thresholds below are stated rather than zero, and each is
/// the measured margin rather than a guess.

#include <viame/image_processing/sift.h>

#include <viame/image_io/codecs/image_codec.h>
#include <viame/core_types/image.h>

#include "../../../tests/golden/golden_json.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using viame::testing::golden_json;

namespace {

std::string
golden_dir()
{
  char const* dir = std::getenv( "VIAME_GOLDEN_SIFT_DIR" );

  if( !dir )
  {
    dir = VIAME_GOLDEN_SIFT_DIR;
  }

  return std::string( dir );
}

/// One recorded keypoint: the six numbers the recorder writes per entry.
struct reference_keypoint
{
  double x, y, size, angle, response;
  int octave;
};

std::vector< reference_keypoint >
reference_keypoints( std::string const& object )
{
  auto const flat = golden_json::numbers( object, "keypoints" );
  std::vector< reference_keypoint > out;

  for( size_t i = 0; i + 5 < flat.size(); i += 6 )
  {
    out.push_back( { flat[ i ], flat[ i + 1 ], flat[ i + 2 ], flat[ i + 3 ],
                     flat[ i + 4 ], static_cast< int >( flat[ i + 5 ] ) } );
  }

  return out;
}

/// The nearest of \p keypoints to \p want, and how far away it is.
///
/// On position, size **and angle**, all three. SIFT emits one keypoint per
/// dominant orientation, so a corner with two strong gradients appears twice at
/// the same place and the same scale; matching on position alone picks whichever
/// came first and reports the other one's angle as an error of nearly 180
/// degrees. Angle is scaled down so that it breaks ties rather than driving the
/// match -- a degree of angle is worth a hundredth of a pixel here.
size_t
nearest( std::vector< viame::sift::keypoint > const& keypoints,
         reference_keypoint const& want, double& distance )
{
  double best = 1e30;
  size_t index = keypoints.size();

  for( size_t k = 0; k < keypoints.size(); ++k )
  {
    auto const dx = keypoints[ k ].x - want.x;
    auto const dy = keypoints[ k ].y - want.y;
    auto const ds = keypoints[ k ].size - want.size;

    auto angle = std::abs( keypoints[ k ].angle - want.angle );
    angle = std::min( angle, 360.0 - angle ) * 0.01;

    auto const cost = dx * dx + dy * dy + ds * ds + angle * angle;

    if( cost < best )
    {
      best = cost;
      index = k;
    }
  }

  distance = std::sqrt( best );

  return index;
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( sift, matches_opencv )
{
  golden_json const g( golden_dir() + "/opencv.json" );
  auto const cases = g.section( "cases" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const tile = golden_json::text( c, "image" );
    auto const name = golden_json::text( c, "tile" ) + " " +
                      golden_json::text( c, "name" );

    auto const loaded = viame::codecs::read( golden_dir() + "/" + tile );
    viame::image_of< uint8_t > const image( loaded );
    ASSERT_EQ( 1u, image.depth() ) << name << ": the tile is grayscale";

    viame::sift::settings settings;
    settings.n_features =
      static_cast< int >( golden_json::number( c, "features" ) );
    settings.n_octave_layers =
      static_cast< int >( golden_json::number( c, "layers" ) );
    settings.contrast_threshold = golden_json::number( c, "contrast" );
    settings.edge_threshold = golden_json::number( c, "edge" );
    settings.sigma = golden_json::number( c, "sigma" );

    std::vector< viame::sift::keypoint > keypoints;
    std::vector< float > descriptors;
    viame::sift::detect_and_compute( image, settings, keypoints, &descriptors );

    auto const expected_count =
      static_cast< size_t >( golden_json::number( c, "count" ) );
    auto const width =
      static_cast< int >( golden_json::number( c, "descriptor_width" ) );

    EXPECT_EQ( width, viame::sift::descriptor_size() )
      << name << ": descriptor width";

    // The count is **equal**, not close. It was written as a tolerance on the
    // expectation that a keypoint whose contrast sits within a float ULP of
    // the threshold could fall either side, and over twelve configurations
    // from 6 keypoints to 1558 not one does.
    EXPECT_EQ( expected_count, keypoints.size() ) << name << ": keypoint count";

    auto const reference = reference_keypoints( c );
    ASSERT_FALSE( reference.empty() ) << name;

    size_t matched = 0;
    double worst_position = 0.0;
    double worst_size = 0.0;
    double worst_angle = 0.0;
    double worst_response = 0.0;

    for( auto const& want : reference )
    {
      double distance = 0.0;
      auto const index = nearest( keypoints, want, distance );

      if( index == keypoints.size() || distance > 1.0 )
      {
        continue;
      }

      auto const& found = keypoints[ index ];

      ++matched;
      worst_position = std::max(
        worst_position, std::sqrt( ( found.x - want.x ) * ( found.x - want.x ) +
                                   ( found.y - want.y ) * ( found.y - want.y ) ) );
      worst_size = std::max( worst_size, std::abs( found.size - want.size ) );

      auto angle = std::abs( found.angle - want.angle );
      angle = std::min( angle, 360.0 - angle );
      worst_angle = std::max( worst_angle, angle );

      auto const denominator = std::max( 1e-6, std::abs( want.response ) );
      worst_response = std::max(
        worst_response,
        std::abs( found.response - want.response ) / denominator );
    }

    auto const rate = static_cast< double >( matched ) /
                      static_cast< double >( reference.size() );

    // Descriptors of the strongest few, by cosine similarity: they are
    // histograms on a 0..255 scale, so the direction is what carries the
    // meaning and the length is a normalisation either side.
    auto const want_descriptors = golden_json::numbers( c, "descriptors" );
    auto const recorded = width > 0 ? want_descriptors.size() /
                                      static_cast< size_t >( width ) : 0;
    double worst_similarity = 1.0;
    size_t compared = 0;

    for( size_t i = 0; i < recorded && i < reference.size(); ++i )
    {
      double distance = 0.0;
      auto const index = nearest( keypoints, reference[ i ], distance );

      if( index == keypoints.size() || distance > 1.0 )
      {
        continue;
      }

      double dot = 0.0;
      double want_norm = 0.0;
      double got_norm = 0.0;

      for( int j = 0; j < width; ++j )
      {
        auto const a = want_descriptors[ i * static_cast< size_t >( width ) +
                                         static_cast< size_t >( j ) ];
        auto const b = static_cast< double >(
          descriptors[ index * static_cast< size_t >( width ) +
                       static_cast< size_t >( j ) ] );

        dot += a * b;
        want_norm += a * a;
        got_norm += b * b;
      }

      if( want_norm > 0.0 && got_norm > 0.0 )
      {
        worst_similarity = std::min(
          worst_similarity, dot / std::sqrt( want_norm * got_norm ) );
        ++compared;
      }
    }

    std::cout << "  " << name << ": " << keypoints.size() << " found against "
              << expected_count << ", " << matched
              << "/" << reference.size() << " matched, worst position "
              << worst_position << " px, size " << worst_size << ", angle "
              << worst_angle << " deg, response " << worst_response
              << ", descriptor similarity " << worst_similarity << " over "
              << compared << "\n";

    // Every threshold here is the measured margin with a little room, not a
    // guess at what SIFT ought to manage. Achieved over the twelve recorded
    // configurations: every keypoint matched, position within 3.0e-05 px, size
    // 1.9e-06, angle 4.6e-05 degrees, response 4.7e-05 relative, and
    // descriptor similarity 0.999998 at worst.
    EXPECT_GE( rate, 1.0 ) << name << ": matched fraction";
    EXPECT_LE( worst_position, 1e-4 ) << name << ": keypoint position";
    EXPECT_LE( worst_size, 1e-5 ) << name << ": keypoint size";
    EXPECT_LE( worst_angle, 1e-3 ) << name << ": keypoint angle";
    EXPECT_LE( worst_response, 1e-4 ) << name << ": keypoint response";
    EXPECT_GE( worst_similarity, 0.99999 ) << name << ": descriptor direction";
  }
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
