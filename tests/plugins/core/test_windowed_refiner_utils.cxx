/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "windowed_utils.h"

#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image.h>
#include <vital/types/image_container.h>

namespace kv = kwiver::vital;
using namespace viame;

// =============================================================================
// Helpers
// =============================================================================

static kv::detected_object_sptr
make_detection( double x1, double y1, double x2, double y2,
                double conf = 0.9 )
{
  auto dot = std::make_shared< kv::detected_object_type >( "fish", conf );
  auto det = std::make_shared< kv::detected_object >(
    kv::bounding_box_d( x1, y1, x2, y2 ), conf, dot );
  return det;
}

static kv::detected_object_sptr
make_detection_with_mask( double x1, double y1, double x2, double y2,
                          double conf = 0.9 )
{
  auto det = make_detection( x1, y1, x2, y2, conf );

  int w = static_cast< int >( x2 - x1 );
  int h = static_cast< int >( y2 - y1 );
  kv::image mask( w, h, 1, false, kv::image_pixel_traits_of< uint8_t >() );
  // Fill entire mask with 1
  for( int y = 0; y < h; y++ )
    for( int x = 0; x < w; x++ )
      mask.at< uint8_t >( x, y ) = 1;

  det->set_mask( std::make_shared< kv::simple_image_container >( mask ) );
  return det;
}

// =============================================================================
// tile_overlap_strip tests
// =============================================================================

TEST( windowed_refiner_utils, tile_overlap_strip_overlapping )
{
  // Two tiles: [0,0,100,100] and [60,0,100,100] overlap in [60,0,40,100]
  image_rect a( 0, 0, 100, 100 );
  image_rect b( 60, 0, 100, 100 );
  int ox, oy, ow, oh;
  ASSERT_TRUE( tile_overlap_strip( a, b, ox, oy, ow, oh ) );
  EXPECT_EQ( ox, 60 );
  EXPECT_EQ( oy, 0 );
  EXPECT_EQ( ow, 40 );
  EXPECT_EQ( oh, 100 );
}

TEST( windowed_refiner_utils, tile_overlap_strip_no_overlap )
{
  image_rect a( 0, 0, 100, 100 );
  image_rect b( 200, 0, 100, 100 );
  int ox, oy, ow, oh;
  ASSERT_FALSE( tile_overlap_strip( a, b, ox, oy, ow, oh ) );
}

TEST( windowed_refiner_utils, tile_overlap_strip_2d )
{
  // 2D overlap: tiles overlap in both x and y
  image_rect a( 0, 0, 100, 100 );
  image_rect b( 60, 50, 100, 100 );
  int ox, oy, ow, oh;
  ASSERT_TRUE( tile_overlap_strip( a, b, ox, oy, ow, oh ) );
  EXPECT_EQ( ox, 60 );
  EXPECT_EQ( oy, 50 );
  EXPECT_EQ( ow, 40 );
  EXPECT_EQ( oh, 50 );
}

// =============================================================================
// render_mask_in_strip tests
// =============================================================================

TEST( windowed_refiner_utils, render_mask_no_mask_uses_bbox )
{
  // Detection at [10,10,30,30] with no mask
  auto det = make_detection( 10, 10, 30, 30 );

  // Strip covers [0,0,50,50] — entire bbox should be filled
  kv::image out;
  int count = render_mask_in_strip( det, 0, 0, 50, 50, out );

  EXPECT_EQ( count, 20 * 20 );  // 20x20 bbox
  EXPECT_EQ( out.width(), 50 );
  EXPECT_EQ( out.height(), 50 );
  // Pixel inside bbox should be 1
  EXPECT_EQ( out.at< uint8_t >( 15, 15 ), 1 );
  // Pixel outside bbox should be 0
  EXPECT_EQ( out.at< uint8_t >( 5, 5 ), 0 );
}

TEST( windowed_refiner_utils, render_mask_with_mask )
{
  auto det = make_detection_with_mask( 10, 10, 30, 30 );

  kv::image out;
  int count = render_mask_in_strip( det, 0, 0, 50, 50, out );

  EXPECT_EQ( count, 20 * 20 );
  EXPECT_EQ( out.at< uint8_t >( 15, 15 ), 1 );
  EXPECT_EQ( out.at< uint8_t >( 5, 5 ), 0 );
}

TEST( windowed_refiner_utils, render_mask_partial_overlap )
{
  // Detection at [10,10,30,30], strip covers [20,20,30,30]
  // Only the [20,20]->[30,30] portion should be rendered
  auto det = make_detection_with_mask( 10, 10, 30, 30 );

  kv::image out;
  int count = render_mask_in_strip( det, 20, 20, 30, 30, out );

  EXPECT_EQ( count, 10 * 10 );  // 10x10 intersection
}

// =============================================================================
// merge_tile_boundary_detections tests
// =============================================================================

TEST( windowed_refiner_utils, merge_same_tile_not_merged )
{
  // Two detections from the same tile should not be merged
  image_rect tile( 0, 0, 100, 100 );

  std::vector< det_tile_entry > entries;
  entries.push_back( { make_detection_with_mask( 10, 10, 30, 30, 0.9 ), tile } );
  entries.push_back( { make_detection_with_mask( 15, 15, 35, 35, 0.8 ), tile } );

  auto result = merge_tile_boundary_detections( entries, 0.5, 200, 200 );
  // Both should remain since they're from the same tile
  EXPECT_EQ( result->size(), 2 );
}

TEST( windowed_refiner_utils, merge_different_tiles_high_overlap )
{
  // Two overlapping tiles with detections that overlap significantly
  // in the tile overlap strip
  image_rect tile_a( 0, 0, 100, 100 );
  image_rect tile_b( 60, 0, 100, 100 );
  // Both detect the same object at roughly the same place
  // in the overlap strip [60,0,40,100]

  std::vector< det_tile_entry > entries;
  entries.push_back( { make_detection_with_mask( 65, 20, 95, 50, 0.9 ), tile_a } );
  entries.push_back( { make_detection_with_mask( 65, 20, 95, 50, 0.8 ), tile_b } );

  auto result = merge_tile_boundary_detections( entries, 0.8, 200, 200 );
  // Should merge into one detection
  EXPECT_EQ( result->size(), 1 );
  // Should keep the higher confidence
  EXPECT_GE( result->begin()->get()->confidence(), 0.9 );
}

TEST( windowed_refiner_utils, merge_different_tiles_low_overlap )
{
  // Detections from different tiles but not overlapping in the strip
  image_rect tile_a( 0, 0, 100, 100 );
  image_rect tile_b( 60, 0, 100, 100 );

  std::vector< det_tile_entry > entries;
  // det_a is in tile_a only (x=10..40, outside the overlap strip [60..100])
  entries.push_back( { make_detection_with_mask( 10, 20, 40, 50, 0.9 ), tile_a } );
  // det_b is in tile_b's non-overlap area (x=110..140)
  entries.push_back( { make_detection_with_mask( 110, 20, 140, 50, 0.8 ), tile_b } );

  auto result = merge_tile_boundary_detections( entries, 0.8, 200, 200 );
  // Should NOT merge — detections don't overlap in the strip
  EXPECT_EQ( result->size(), 2 );
}

TEST( windowed_refiner_utils, merge_preserves_union_bbox )
{
  image_rect tile_a( 0, 0, 100, 100 );
  image_rect tile_b( 60, 0, 100, 100 );

  // Detection from tile_a extends from 50 to 90
  // Detection from tile_b extends from 70 to 120
  // They overlap in the tile strip [60,100] at [70,90]
  std::vector< det_tile_entry > entries;
  entries.push_back( { make_detection_with_mask( 50, 20, 90, 50, 0.9 ), tile_a } );
  entries.push_back( { make_detection_with_mask( 70, 20, 120, 50, 0.8 ), tile_b } );

  auto result = merge_tile_boundary_detections( entries, 0.5, 200, 200 );
  ASSERT_EQ( result->size(), 1 );

  auto merged = result->begin()->get();
  auto bb = merged->bounding_box();
  // Union bbox should span 50 to 120
  EXPECT_LE( bb.min_x(), 50.0 );
  EXPECT_GE( bb.max_x(), 120.0 );
}

TEST( windowed_refiner_utils, merge_non_overlapping_tiles )
{
  // Tiles that don't overlap at all — no merging possible
  image_rect tile_a( 0, 0, 100, 100 );
  image_rect tile_b( 200, 0, 100, 100 );

  std::vector< det_tile_entry > entries;
  entries.push_back( { make_detection_with_mask( 10, 10, 30, 30, 0.9 ), tile_a } );
  entries.push_back( { make_detection_with_mask( 210, 10, 230, 30, 0.8 ), tile_b } );

  auto result = merge_tile_boundary_detections( entries, 0.8, 400, 200 );
  EXPECT_EQ( result->size(), 2 );
}
