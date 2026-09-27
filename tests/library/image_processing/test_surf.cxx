/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The SURF port against what OpenCV computes.
///
/// `tests/golden/surf/opencv.json` is recorded by
/// `tests/golden/surf/record_from_opencv.py` under a cv2 built with the
/// non-free modules -- this branch's wheel is not one, which is why the
/// algorithm is ported at all. The tiles it was recorded on are saved beside
/// it as PNGs and read here, so the grayscale conversion is not a variable.
///
/// This is a port rather than a reimplementation, so the bar is agreement,
/// not similarity: the same keypoints, in the same places, with descriptors
/// that point the same way.

#include <viame/image_processing/surf.h>

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
  char const* dir = std::getenv( "VIAME_GOLDEN_SURF_DIR" );

  if( !dir )
  {
    dir = VIAME_GOLDEN_SURF_DIR;
  }

  return std::string( dir );
}

/// One recorded keypoint: the seven numbers the recorder writes per entry.
struct reference_keypoint
{
  double x, y, size, angle, response;
  int octave, laplacian;
};

std::vector< reference_keypoint >
reference_keypoints( std::string const& object )
{
  auto const flat = golden_json::numbers( object, "keypoints" );
  std::vector< reference_keypoint > out;

  for( size_t i = 0; i + 6 < flat.size(); i += 7 )
  {
    out.push_back( { flat[ i ], flat[ i + 1 ], flat[ i + 2 ], flat[ i + 3 ],
                     flat[ i + 4 ], static_cast< int >( flat[ i + 5 ] ),
                     static_cast< int >( flat[ i + 6 ] ) } );
  }

  return out;
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( surf, matches_opencv )
{
  golden_json const g( golden_dir() + "/opencv.json" );
  auto const cases = g.section( "cases" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const tile = golden_json::text( c, "image" );
    auto const name = tile + " " + golden_json::text( c, "name" );

    auto const loaded = viame::codecs::read( golden_dir() + "/" + tile );
    viame::image_of< uint8_t > const image( loaded );
    ASSERT_EQ( 1u, image.depth() ) << name << ": the tile is grayscale";

    viame::surf::settings settings;
    settings.hessian_threshold = golden_json::number( c, "hessian" );
    settings.n_octaves = static_cast< int >( golden_json::number( c, "octaves" ) );
    settings.n_octaves_layers =
      static_cast< int >( golden_json::number( c, "layers" ) );
    settings.extended = golden_json::number( c, "extended" ) != 0;
    settings.upright = golden_json::number( c, "upright" ) != 0;

    std::vector< viame::surf::keypoint > keypoints;
    std::vector< float > descriptors;
    viame::surf::detect_and_compute( image, settings, keypoints, &descriptors );

    auto const expected_count =
      static_cast< size_t >( golden_json::number( c, "count" ) );
    auto const width =
      static_cast< int >( golden_json::number( c, "descriptor_width" ) );

    EXPECT_EQ( expected_count, keypoints.size() ) << name << ": keypoint count";
    EXPECT_EQ( width, viame::surf::descriptor_size( settings ) )
      << name << ": descriptor width";

    auto const reference = reference_keypoints( c );
    ASSERT_FALSE( reference.empty() ) << name;

    // Match by position: the recorded order is by response, and ties in it
    // are not ordered the same way on both sides.
    size_t matched = 0;
    double worst_position = 0.0;
    double worst_angle = 0.0;
    double worst_response = 0.0;

    for( auto const& want : reference )
    {
      double best = 1e30;
      viame::surf::keypoint const* found = nullptr;

      for( auto const& got : keypoints )
      {
        double const dx = got.x - want.x;
        double const dy = got.y - want.y;
        double const d = dx * dx + dy * dy;

        if( d < best )
        {
          best = d;
          found = &got;
        }
      }

      if( !found || std::sqrt( best ) > 1.0 )
      {
        continue;
      }

      ++matched;
      worst_position = std::max( worst_position, std::sqrt( best ) );

      // Angles wrap, so compare the shorter way round.
      double angle = std::abs( found->angle - want.angle );
      angle = std::min( angle, 360.0 - angle );
      worst_angle = std::max( worst_angle, angle );

      double const denominator = std::max( 1.0, std::abs( want.response ) );
      worst_response = std::max(
        worst_response, std::abs( found->response - want.response ) / denominator );

      EXPECT_EQ( want.laplacian, found->laplacian ) << name << ": laplacian";
    }

    double const rate =
      static_cast< double >( matched ) / static_cast< double >( reference.size() );

    std::cout << "  " << name << ": " << keypoints.size() << " found against "
              << expected_count << ", " << matched << "/" << reference.size()
              << " matched, worst position " << worst_position
              << " px, angle " << worst_angle << " deg, response "
              << worst_response << "\n";

    EXPECT_GE( rate, 0.98 ) << name << ": matched fraction";
    EXPECT_LE( worst_position, 0.05 ) << name << ": keypoint position";
    EXPECT_LE( worst_response, 1e-4 ) << name << ": keypoint response";

    // The descriptors of the strongest few, compared by direction: they are
    // unit vectors, so a dot product is the whole story.
    auto const want_descriptors = golden_json::numbers( c, "descriptors" );
    size_t const recorded = want_descriptors.size() / width;
    double worst_similarity = 1.0;

    for( size_t i = 0; i < recorded && i < reference.size(); ++i )
    {
      // Find our keypoint for this reference one again.
      double best = 1e30;
      size_t index = 0;

      for( size_t k = 0; k < keypoints.size(); ++k )
      {
        double const dx = keypoints[ k ].x - reference[ i ].x;
        double const dy = keypoints[ k ].y - reference[ i ].y;
        double const d = dx * dx + dy * dy;

        if( d < best ) { best = d; index = k; }
      }

      if( std::sqrt( best ) > 1.0 ) { continue; }

      double dot = 0.0;
      for( int j = 0; j < width; ++j )
      {
        dot += want_descriptors[ i * width + j ] *
               descriptors[ index * static_cast< size_t >( width ) + j ];
      }

      worst_similarity = std::min( worst_similarity, dot );
    }

    std::cout << "    descriptor similarity, worst of " << recorded << ": "
              << worst_similarity << "\n";

    EXPECT_GE( worst_similarity, 0.99 ) << name << ": descriptor direction";
  }
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
