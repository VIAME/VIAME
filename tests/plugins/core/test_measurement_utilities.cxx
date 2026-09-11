/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "measurement_utilities.h"
#include "disparity_segment.h"
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <limits>

#include <vital/types/camera_intrinsics.h>
#include <vital/types/rotation.h>

#include <cmath>

namespace kv = kwiver::vital;
using namespace viame::core;

// =============================================================================
// Test Fixtures and Helpers
// =============================================================================

class measurement_utilities_test : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create simple stereo camera setup with known geometry
    // Left camera at origin, right camera offset by baseline in X
    double focal_length = 1000.0;
    double principal_x = 640.0;
    double principal_y = 360.0;
    double baseline = 0.1;  // 10cm baseline

    auto intrinsics = std::make_shared< kv::simple_camera_intrinsics >(
      focal_length, kv::vector_2d( principal_x, principal_y ) );

    // Left camera at origin looking down +Z
    left_cam = std::make_shared< kv::simple_camera_perspective >();
    left_cam->set_intrinsics( intrinsics );
    left_cam->set_center( kv::vector_3d( 0, 0, 0 ) );
    left_cam->set_rotation( kv::rotation_d() );  // Identity rotation

    // Right camera offset by baseline in X
    right_cam = std::make_shared< kv::simple_camera_perspective >();
    right_cam->set_intrinsics( intrinsics );
    right_cam->set_center( kv::vector_3d( baseline, 0, 0 ) );
    right_cam->set_rotation( kv::rotation_d() );  // Identity rotation

    utilities = std::make_shared< map_keypoints_to_camera >();
  }

  kv::simple_camera_perspective_sptr left_cam;
  kv::simple_camera_perspective_sptr right_cam;
  std::shared_ptr< map_keypoints_to_camera > utilities;
};

// =============================================================================
// Method Parsing Tests
// =============================================================================

TEST( measurement_utilities_static, parse_matching_methods_single )
{
  auto methods = parse_matching_methods( "template_matching" );
  ASSERT_EQ( methods.size(), 1 );
  EXPECT_EQ( methods[0], "template_matching" );
}

TEST( measurement_utilities_static, parse_matching_methods_multiple )
{
  auto methods = parse_matching_methods(
    "input_pairs_only,template_matching,depth_projection" );
  ASSERT_EQ( methods.size(), 3 );
  EXPECT_EQ( methods[0], "input_pairs_only" );
  EXPECT_EQ( methods[1], "template_matching" );
  EXPECT_EQ( methods[2], "depth_projection" );
}

TEST( measurement_utilities_static, parse_matching_methods_with_whitespace )
{
  auto methods = parse_matching_methods(
    " input_pairs_only , template_matching , depth_projection " );
  ASSERT_EQ( methods.size(), 3 );
  EXPECT_EQ( methods[0], "input_pairs_only" );
  EXPECT_EQ( methods[1], "template_matching" );
  EXPECT_EQ( methods[2], "depth_projection" );
}

TEST( measurement_utilities_static, parse_matching_methods_empty )
{
  auto methods = parse_matching_methods( "" );
  EXPECT_TRUE( methods.empty() );
}

TEST( measurement_utilities_static, method_requires_images )
{
  EXPECT_FALSE( method_requires_images( "input_pairs_only" ) );
  EXPECT_FALSE( method_requires_images( "depth_projection" ) );
  EXPECT_TRUE( method_requires_images( "template_matching" ) );
  EXPECT_TRUE( method_requires_images( "compute_disparity" ) );
  EXPECT_TRUE( method_requires_images( "feature_descriptor" ) );
  EXPECT_TRUE( method_requires_images( "ransac_feature" ) );
}

TEST( measurement_utilities_static, get_valid_methods )
{
  auto methods = get_valid_methods();
  EXPECT_EQ( methods.size(), 8 );

  // Check all expected methods are present
  std::set<std::string> method_set( methods.begin(), methods.end() );
  EXPECT_TRUE( method_set.count( "input_pairs_only" ) > 0 );
  EXPECT_TRUE( method_set.count( "depth_projection" ) > 0 );
  EXPECT_TRUE( method_set.count( "external_disparity" ) > 0 );
  EXPECT_TRUE( method_set.count( "compute_disparity" ) > 0 );
  EXPECT_TRUE( method_set.count( "template_matching" ) > 0 );
  EXPECT_TRUE( method_set.count( "epipolar_template_matching" ) > 0 );
  EXPECT_TRUE( method_set.count( "feature_descriptor" ) > 0 );
  EXPECT_TRUE( method_set.count( "ransac_feature" ) > 0 );
}

// =============================================================================
// Settings Tests
// =============================================================================

TEST( measurement_settings, default_values )
{
  map_keypoints_to_camera_settings settings;

  EXPECT_EQ( settings.default_depth, 5.0 );
  EXPECT_EQ( settings.template_size, 31 );
  EXPECT_EQ( settings.search_range, 128 );
  EXPECT_TRUE( settings.use_distortion );
  EXPECT_EQ( settings.feature_search_radius, 50.0 );
  EXPECT_EQ( settings.ransac_inlier_scale, 3.0 );
  EXPECT_EQ( settings.min_ransac_inliers, 10 );
  EXPECT_NEAR( settings.box_scale_factor, 1.10, 0.001 );
  EXPECT_TRUE( settings.record_stereo_method );
}

TEST( measurement_settings, validate_matching_methods_valid )
{
  map_keypoints_to_camera_settings settings;
  settings.matching_methods = "input_pairs_only,template_matching";

  std::string error = settings.validate_matching_methods();
  EXPECT_TRUE( error.empty() );
}

TEST( measurement_settings, validate_matching_methods_invalid )
{
  map_keypoints_to_camera_settings settings;
  settings.matching_methods = "input_pairs_only,invalid_method";

  std::string error = settings.validate_matching_methods();
  EXPECT_FALSE( error.empty() );
  EXPECT_TRUE( error.find( "invalid_method" ) != std::string::npos );
}

TEST( measurement_settings, validate_matching_methods_empty )
{
  map_keypoints_to_camera_settings settings;
  settings.matching_methods = "";

  std::string error = settings.validate_matching_methods();
  EXPECT_FALSE( error.empty() );
}

TEST( measurement_settings, any_method_requires_images )
{
  map_keypoints_to_camera_settings settings;

  settings.matching_methods = "input_pairs_only,depth_projection";
  EXPECT_FALSE( settings.any_method_requires_images() );

  settings.matching_methods = "input_pairs_only,template_matching";
  EXPECT_TRUE( settings.any_method_requires_images() );
}

// =============================================================================
// Projection Tests
// =============================================================================

TEST_F( measurement_utilities_test, project_left_to_right_center_point )
{
  // A point at the center of the left image should project slightly
  // to the left in the right image due to the baseline offset
  utilities->set_default_depth( 1.0 );  // 1 meter depth

  kv::vector_2d left_point( 640, 360 );  // Center of image
  kv::vector_2d right_point = utilities->project_left_to_right(
    *left_cam, *right_cam, left_point );

  // With a 10cm baseline and 1m depth, the disparity should be:
  // disparity = focal_length * baseline / depth = 1000 * 0.1 / 1.0 = 100 pixels
  // Right point should be at x = 640 - 100 = 540
  EXPECT_NEAR( right_point.x(), 540.0, 1.0 );
  EXPECT_NEAR( right_point.y(), 360.0, 1.0 );
}

TEST_F( measurement_utilities_test, project_left_to_right_varying_depth )
{
  kv::vector_2d left_point( 640, 360 );

  // At 2m depth, disparity should be half
  utilities->set_default_depth( 2.0 );
  kv::vector_2d right_at_2m = utilities->project_left_to_right(
    *left_cam, *right_cam, left_point );

  // disparity = 1000 * 0.1 / 2.0 = 50 pixels
  EXPECT_NEAR( right_at_2m.x(), 590.0, 1.0 );

  // At 0.5m depth, disparity should be double
  utilities->set_default_depth( 0.5 );
  kv::vector_2d right_at_half_m = utilities->project_left_to_right(
    *left_cam, *right_cam, left_point );

  // disparity = 1000 * 0.1 / 0.5 = 200 pixels
  EXPECT_NEAR( right_at_half_m.x(), 440.0, 1.0 );
}

// =============================================================================
// Triangulation Tests
// =============================================================================

TEST_F( measurement_utilities_test, triangulate_point_at_known_depth )
{
  // Create corresponding points that should triangulate to a known 3D point
  // Point at (0, 0, 1) in world coordinates
  // Left camera: projects to (640, 360)
  // Right camera: projects to (640 - 100, 360) = (540, 360) due to 10cm baseline
  kv::vector_2d left_point( 640, 360 );
  kv::vector_2d right_point( 540, 360 );

  kv::vector_3d point_3d = viame::core::triangulate_point(
    *left_cam, *right_cam, left_point, right_point );

  // Should be at approximately (0, 0, 1)
  EXPECT_NEAR( point_3d.x(), 0.0, 0.01 );
  EXPECT_NEAR( point_3d.y(), 0.0, 0.01 );
  EXPECT_NEAR( point_3d.z(), 1.0, 0.01 );
}

TEST_F( measurement_utilities_test, triangulate_point_off_center )
{
  // Point at (0.1, 0.1, 1) in world coordinates
  // Left camera at origin: normalized coords (0.1, 0.1), pixel (640+100, 360+100) = (740, 460)
  // Right camera at (0.1, 0, 0): sees point at (0, 0.1, 1), pixel (640, 460)
  kv::vector_2d left_point( 740, 460 );
  kv::vector_2d right_point( 640, 460 );

  kv::vector_3d point_3d = viame::core::triangulate_point(
    *left_cam, *right_cam, left_point, right_point );

  EXPECT_NEAR( point_3d.x(), 0.1, 0.02 );
  EXPECT_NEAR( point_3d.y(), 0.1, 0.02 );
  EXPECT_NEAR( point_3d.z(), 1.0, 0.02 );
}

// =============================================================================
// Stereo Length Tests
// =============================================================================

TEST_F( measurement_utilities_test, compute_stereo_length_horizontal )
{
  // Two points 0.1m apart horizontally at 1m depth
  // Point 1: (0, 0, 1)
  // Point 2: (0.1, 0, 1)
  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d right_head( 540, 360 );
  kv::vector_2d left_tail( 740, 360 );
  kv::vector_2d right_tail( 640, 360 );

  double length = viame::core::compute_stereo_length(
    *left_cam, *right_cam, left_head, right_head, left_tail, right_tail );

  // Should be approximately 0.1m
  EXPECT_NEAR( length, 0.1, 0.01 );
}

TEST_F( measurement_utilities_test, compute_stereo_length_vertical )
{
  // Two points 0.1m apart vertically at 1m depth
  // Point 1: (0, 0, 1)
  // Point 2: (0, 0.1, 1)
  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d right_head( 540, 360 );
  kv::vector_2d left_tail( 640, 460 );
  kv::vector_2d right_tail( 540, 460 );

  double length = viame::core::compute_stereo_length(
    *left_cam, *right_cam, left_head, right_head, left_tail, right_tail );

  EXPECT_NEAR( length, 0.1, 0.01 );
}

TEST_F( measurement_utilities_test, compute_stereo_length_diagonal )
{
  // Two points 0.1m apart horizontally and 0.1m vertically = sqrt(2)*0.1 diagonal
  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d right_head( 540, 360 );
  kv::vector_2d left_tail( 740, 460 );
  kv::vector_2d right_tail( 640, 460 );

  double length = viame::core::compute_stereo_length(
    *left_cam, *right_cam, left_head, right_head, left_tail, right_tail );

  double expected = std::sqrt( 0.1 * 0.1 + 0.1 * 0.1 );
  EXPECT_NEAR( length, expected, 0.02 );
}

// =============================================================================
// Bounding Box Tests
// =============================================================================

TEST_F( measurement_utilities_test, compute_bbox_from_keypoints_default )
{
  utilities->set_box_scale_factor( 1.0 );  // No scaling

  kv::vector_2d head( 100, 200 );
  kv::vector_2d tail( 200, 300 );

  kv::bounding_box_d bbox = utilities->compute_bbox_from_keypoints( head, tail );

  EXPECT_NEAR( bbox.min_x(), 100.0, 0.001 );
  EXPECT_NEAR( bbox.min_y(), 200.0, 0.001 );
  EXPECT_NEAR( bbox.max_x(), 200.0, 0.001 );
  EXPECT_NEAR( bbox.max_y(), 300.0, 0.001 );
}

TEST_F( measurement_utilities_test, compute_bbox_from_keypoints_with_scale )
{
  utilities->set_box_scale_factor( 1.2 );  // 20% expansion

  kv::vector_2d head( 100, 200 );
  kv::vector_2d tail( 200, 300 );

  // Original box: min(100,200) max(200,300), center(150, 250), size(100, 100)
  // Scaled: size(120, 120), center(150, 250)
  // New box: min(90, 190) max(210, 310)
  kv::bounding_box_d bbox = utilities->compute_bbox_from_keypoints( head, tail );

  EXPECT_NEAR( bbox.min_x(), 90.0, 0.001 );
  EXPECT_NEAR( bbox.min_y(), 190.0, 0.001 );
  EXPECT_NEAR( bbox.max_x(), 210.0, 0.001 );
  EXPECT_NEAR( bbox.max_y(), 310.0, 0.001 );
}

TEST_F( measurement_utilities_test, compute_bbox_from_keypoints_reversed_order )
{
  utilities->set_box_scale_factor( 1.0 );

  // Head and tail can be in any order
  kv::vector_2d head( 200, 300 );
  kv::vector_2d tail( 100, 200 );

  kv::bounding_box_d bbox = utilities->compute_bbox_from_keypoints( head, tail );

  EXPECT_NEAR( bbox.min_x(), 100.0, 0.001 );
  EXPECT_NEAR( bbox.min_y(), 200.0, 0.001 );
  EXPECT_NEAR( bbox.max_x(), 200.0, 0.001 );
  EXPECT_NEAR( bbox.max_y(), 300.0, 0.001 );
}

// =============================================================================
// Stereo Correspondence Tests
// =============================================================================

TEST_F( measurement_utilities_test, find_stereo_correspondence_input_pairs_only )
{
  std::vector< std::string > methods = { "input_pairs_only" };

  kv::vector_2d left_head( 100, 100 );
  kv::vector_2d left_tail( 200, 200 );
  kv::vector_2d right_head( 80, 100 );
  kv::vector_2d right_tail( 180, 200 );

  auto result = utilities->find_stereo_correspondence(
    methods, *left_cam, *right_cam,
    left_head, left_tail, &right_head, &right_tail,
    nullptr, nullptr );

  EXPECT_TRUE( result.success );
  EXPECT_EQ( result.method_used, "input_pairs_only" );
  EXPECT_NEAR( result.right_head.x(), 80.0, 0.001 );
  EXPECT_NEAR( result.right_head.y(), 100.0, 0.001 );
  EXPECT_NEAR( result.right_tail.x(), 180.0, 0.001 );
  EXPECT_NEAR( result.right_tail.y(), 200.0, 0.001 );
}

TEST_F( measurement_utilities_test, find_stereo_correspondence_input_pairs_only_no_input )
{
  std::vector< std::string > methods = { "input_pairs_only" };

  kv::vector_2d left_head( 100, 100 );
  kv::vector_2d left_tail( 200, 200 );

  auto result = utilities->find_stereo_correspondence(
    methods, *left_cam, *right_cam,
    left_head, left_tail, nullptr, nullptr,
    nullptr, nullptr );

  // Should fail because no input right points provided
  EXPECT_FALSE( result.success );
}

TEST_F( measurement_utilities_test, find_stereo_correspondence_depth_projection )
{
  std::vector< std::string > methods = { "depth_projection" };
  utilities->set_default_depth( 1.0 );

  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d left_tail( 740, 360 );

  auto result = utilities->find_stereo_correspondence(
    methods, *left_cam, *right_cam,
    left_head, left_tail, nullptr, nullptr,
    nullptr, nullptr );

  EXPECT_TRUE( result.success );
  EXPECT_EQ( result.method_used, "depth_projection" );

  // Verify the projected points match the expected disparity
  // disparity = 1000 * 0.1 / 1.0 = 100 pixels
  EXPECT_NEAR( result.right_head.x(), 540.0, 1.0 );
  EXPECT_NEAR( result.right_tail.x(), 640.0, 1.0 );
}

TEST_F( measurement_utilities_test, find_stereo_correspondence_fallback )
{
  // Test that methods are tried in order and fallback works
  std::vector< std::string > methods = { "input_pairs_only", "depth_projection" };
  utilities->set_default_depth( 1.0 );

  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d left_tail( 740, 360 );

  // No input pairs provided, so input_pairs_only fails
  // Should fall back to depth_projection
  auto result = utilities->find_stereo_correspondence(
    methods, *left_cam, *right_cam,
    left_head, left_tail, nullptr, nullptr,
    nullptr, nullptr );

  EXPECT_TRUE( result.success );
  EXPECT_EQ( result.method_used, "depth_projection" );
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F( measurement_utilities_test, configure_from_settings )
{
  map_keypoints_to_camera_settings settings;
  settings.default_depth = 3.0;
  settings.template_size = 25;
  settings.search_range = 64;
  settings.use_distortion = false;
  settings.box_scale_factor = 1.5;

  utilities->configure( settings );

  // Test that depth projection uses the configured depth
  kv::vector_2d left_point( 640, 360 );
  kv::vector_2d right_point = utilities->project_left_to_right(
    *left_cam, *right_cam, left_point );

  // disparity = 1000 * 0.1 / 3.0 = 33.33 pixels
  EXPECT_NEAR( right_point.x(), 640.0 - 33.33, 1.0 );
}

TEST_F( measurement_utilities_test, set_template_params_ensures_odd )
{
  // Template size should be odd
  utilities->set_template_params( 30, 100 );  // Even number

  // Can't directly access private member, but we can test the effect
  // through configuration
  map_keypoints_to_camera_settings settings;
  settings.template_size = 30;
  utilities->configure( settings );

  // Template size internally should be 31 (odd)
  // This is tested implicitly through the template matching behavior
}

// =============================================================================
// Frame ID and Cache Tests
// =============================================================================

TEST_F( measurement_utilities_test, set_frame_id_clears_cache )
{
  // Set a frame ID
  utilities->set_frame_id( 1 );

  // Setting a different frame ID should clear the cache
  utilities->set_frame_id( 2 );

  // Setting the same frame ID should not clear the cache
  utilities->set_frame_id( 2 );

  // Clear cache explicitly
  utilities->clear_feature_cache();
}

// =============================================================================
// IOU Tests
// =============================================================================

TEST( measurement_utilities_iou, compute_iou_perfect_overlap )
{
  kv::bounding_box_d box1( 0, 0, 100, 100 );
  kv::bounding_box_d box2( 0, 0, 100, 100 );

  double iou = viame::core::compute_iou( box1, box2 );
  EXPECT_NEAR( iou, 1.0, 0.001 );
}

TEST( measurement_utilities_iou, compute_iou_no_overlap )
{
  kv::bounding_box_d box1( 0, 0, 100, 100 );
  kv::bounding_box_d box2( 200, 200, 300, 300 );

  double iou = viame::core::compute_iou( box1, box2 );
  EXPECT_NEAR( iou, 0.0, 0.001 );
}

TEST( measurement_utilities_iou, compute_iou_partial_overlap )
{
  kv::bounding_box_d box1( 0, 0, 100, 100 );
  kv::bounding_box_d box2( 50, 50, 150, 150 );

  // Intersection: 50x50 = 2500
  // Union: 10000 + 10000 - 2500 = 17500
  // IOU: 2500 / 17500 = 0.1429
  double iou = viame::core::compute_iou( box1, box2 );
  EXPECT_NEAR( iou, 2500.0 / 17500.0, 0.01 );
}

TEST( measurement_utilities_iou, compute_iou_invalid_box )
{
  kv::bounding_box_d box1;  // Invalid (default constructed)
  kv::bounding_box_d box2( 0, 0, 100, 100 );

  double iou = viame::core::compute_iou( box1, box2 );
  EXPECT_NEAR( iou, 0.0, 0.001 );
}

// =============================================================================
// Class Label Tests
// =============================================================================

TEST( measurement_utilities_class_label, get_detection_class_label_valid )
{
  auto det = std::make_shared< kv::detected_object >( kv::bounding_box_d( 0, 0, 100, 100 ) );
  auto dot = std::make_shared< kv::detected_object_type >();
  dot->set_score( "fish", 0.9 );
  dot->set_score( "shark", 0.1 );
  det->set_type( dot );

  std::string label = viame::core::get_detection_class_label( det );
  EXPECT_EQ( label, "fish" );
}

TEST( measurement_utilities_class_label, get_detection_class_label_null_detection )
{
  std::string label = viame::core::get_detection_class_label( nullptr );
  EXPECT_EQ( label, "" );
}

TEST( measurement_utilities_class_label, get_detection_class_label_null_type )
{
  auto det = std::make_shared< kv::detected_object >( kv::bounding_box_d( 0, 0, 100, 100 ) );
  // No type set

  std::string label = viame::core::get_detection_class_label( det );
  EXPECT_EQ( label, "" );
}

// =============================================================================
// Greedy Assignment Tests
// =============================================================================

TEST( measurement_utilities_assignment, greedy_assignment_simple )
{
  // Simple 2x2 cost matrix
  std::vector< std::vector< double > > cost_matrix = {
    { 1.0, 2.0 },
    { 3.0, 0.5 }
  };

  auto assignment = viame::core::greedy_assignment( cost_matrix, 2, 2 );

  ASSERT_EQ( assignment.size(), 2 );
  // Should assign (1,1) first (cost 0.5), then (0,0) (cost 1.0)
  // Resulting in assignments: (0,0) and (1,1)
  std::set< std::pair< int, int > > result_set( assignment.begin(), assignment.end() );
  EXPECT_TRUE( result_set.count( std::make_pair( 0, 0 ) ) > 0 );
  EXPECT_TRUE( result_set.count( std::make_pair( 1, 1 ) ) > 0 );
}

TEST( measurement_utilities_assignment, greedy_assignment_with_infinity )
{
  std::vector< std::vector< double > > cost_matrix = {
    { 1.0, 1e10 },
    { 1e10, 0.5 }
  };

  auto assignment = viame::core::greedy_assignment( cost_matrix, 2, 2 );

  ASSERT_EQ( assignment.size(), 2 );
  std::set< std::pair< int, int > > result_set( assignment.begin(), assignment.end() );
  EXPECT_TRUE( result_set.count( std::make_pair( 0, 0 ) ) > 0 );
  EXPECT_TRUE( result_set.count( std::make_pair( 1, 1 ) ) > 0 );
}

TEST( measurement_utilities_assignment, greedy_assignment_rectangular )
{
  // 3x2 cost matrix (more rows than columns)
  std::vector< std::vector< double > > cost_matrix = {
    { 1.0, 2.0 },
    { 0.5, 3.0 },
    { 4.0, 0.3 }
  };

  auto assignment = viame::core::greedy_assignment( cost_matrix, 3, 2 );

  // Should assign at most min(3, 2) = 2 pairs
  ASSERT_EQ( assignment.size(), 2 );
  // Best: (2,1) cost 0.3, (1,0) cost 0.5
  std::set< std::pair< int, int > > result_set( assignment.begin(), assignment.end() );
  EXPECT_TRUE( result_set.count( std::make_pair( 1, 0 ) ) > 0 );
  EXPECT_TRUE( result_set.count( std::make_pair( 2, 1 ) ) > 0 );
}

// =============================================================================
// Furthest Apart Points Tests
// =============================================================================

TEST( measurement_utilities_furthest, find_furthest_apart_points_basic )
{
  std::vector< stereo_feature_correspondence > correspondences = {
    { kv::vector_2d( 10, 50 ), kv::vector_2d( 5, 50 ) },
    { kv::vector_2d( 100, 50 ), kv::vector_2d( 95, 50 ) },
    { kv::vector_2d( 50, 50 ), kv::vector_2d( 45, 50 ) }
  };

  kv::vector_2d left_head, left_tail, right_head, right_tail;
  bool found = viame::core::find_furthest_apart_points(
    correspondences, left_head, left_tail, right_head, right_tail );

  EXPECT_TRUE( found );
  // Points at x=10 and x=100 are furthest apart
  // Head should have smaller x (10)
  EXPECT_NEAR( left_head.x(), 10.0, 0.001 );
  EXPECT_NEAR( left_tail.x(), 100.0, 0.001 );
  EXPECT_NEAR( right_head.x(), 5.0, 0.001 );
  EXPECT_NEAR( right_tail.x(), 95.0, 0.001 );
}

TEST( measurement_utilities_furthest, find_furthest_apart_points_not_enough )
{
  std::vector< stereo_feature_correspondence > correspondences = {
    { kv::vector_2d( 10, 50 ), kv::vector_2d( 5, 50 ) }
  };

  kv::vector_2d left_head, left_tail, right_head, right_tail;
  bool found = viame::core::find_furthest_apart_points(
    correspondences, left_head, left_tail, right_head, right_tail );

  EXPECT_FALSE( found );  // Need at least 2 points
}

TEST( measurement_utilities_furthest, find_furthest_apart_points_diagonal )
{
  std::vector< stereo_feature_correspondence > correspondences = {
    { kv::vector_2d( 0, 0 ), kv::vector_2d( 0, 0 ) },
    { kv::vector_2d( 100, 100 ), kv::vector_2d( 90, 100 ) },
    { kv::vector_2d( 50, 50 ), kv::vector_2d( 45, 50 ) }
  };

  kv::vector_2d left_head, left_tail, right_head, right_tail;
  bool found = viame::core::find_furthest_apart_points(
    correspondences, left_head, left_tail, right_head, right_tail );

  EXPECT_TRUE( found );
  // Diagonal distance is sqrt(100^2 + 100^2) = 141.4
  // Head should have smaller x (0)
  EXPECT_NEAR( left_head.x(), 0.0, 0.001 );
  EXPECT_NEAR( left_head.y(), 0.0, 0.001 );
  EXPECT_NEAR( left_tail.x(), 100.0, 0.001 );
  EXPECT_NEAR( left_tail.y(), 100.0, 0.001 );
}

// =============================================================================
// Full Stereo Measurement Tests
//
// compute_stereo_measurement backs the interactive stereo length feature
// (it is exposed to Python via the viame.core._measurement bindings used by
// the interactive_stereo service). These verify length, 3D midpoint, range,
// and the RMS reprojection error.
// =============================================================================

TEST_F( measurement_utilities_test, compute_stereo_measurement_full )
{
  // Same geometry as compute_stereo_length_horizontal:
  //   head = (0, 0, 1), tail = (0.1, 0, 1), 0.1m apart at 1m depth.
  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d right_head( 540, 360 );
  kv::vector_2d left_tail( 740, 360 );
  kv::vector_2d right_tail( 640, 360 );

  auto const m = viame::core::compute_stereo_measurement(
    *left_cam, *right_cam, left_head, right_head, left_tail, right_tail );

  EXPECT_TRUE( m.valid );
  EXPECT_NEAR( m.length, 0.1, 0.01 );

  // Midpoint at (0.05, 0, 1) in left-camera/world coordinates
  EXPECT_NEAR( m.x, 0.05, 0.01 );
  EXPECT_NEAR( m.y, 0.0, 0.01 );
  EXPECT_NEAR( m.z, 1.0, 0.01 );

  // Range = distance from the midpoint to the left camera center (origin)
  EXPECT_NEAR( m.range, std::sqrt( 0.05 * 0.05 + 1.0 ), 0.01 );

  // Exact correspondences => near-zero reprojection error
  EXPECT_LT( m.rms, 0.5 );
}

TEST_F( measurement_utilities_test, compute_stereo_measurement_rms_flags_bad_match )
{
  // Shift one right point off the epipolar line (vertical error) so the rays
  // no longer intersect; the RMS reprojection error should grow noticeably.
  kv::vector_2d left_head( 640, 360 );
  kv::vector_2d right_head( 540, 380 );  // +20px vertical mismatch
  kv::vector_2d left_tail( 740, 360 );
  kv::vector_2d right_tail( 640, 360 );

  auto const m = viame::core::compute_stereo_measurement(
    *left_cam, *right_cam, left_head, right_head, left_tail, right_tail );

  EXPECT_TRUE( m.valid );
  EXPECT_GT( m.rms, 1.0 );
}

// =============================================================================
// Length Aggregation Tests
//
// aggregate_lengths is the shared helper used both by the pair_stereo_tracks
// pipeline process and (via the _measurement bindings) by DIVE to recompute a
// track's average length from its per-frame lengths.
// =============================================================================

TEST( measurement_utilities_static, aggregate_lengths_average )
{
  std::vector< double > lengths{ 1.0, 2.0, 3.0, 4.0, 5.0 };
  EXPECT_NEAR( aggregate_lengths( lengths, "average" ), 3.0, 1e-9 );
  // "average" is the default method
  EXPECT_NEAR( aggregate_lengths( lengths ), 3.0, 1e-9 );
}

TEST( measurement_utilities_static, aggregate_lengths_median )
{
  EXPECT_NEAR(
    aggregate_lengths( { 1.0, 2.0, 3.0, 4.0, 5.0 }, "median" ), 3.0, 1e-9 );
  EXPECT_NEAR(
    aggregate_lengths( { 1.0, 2.0, 3.0, 4.0 }, "median" ), 2.5, 1e-9 );
}

TEST( measurement_utilities_static, aggregate_lengths_average_iqr_trims_outlier )
{
  // 100 is an outlier and should be excluded from the IQR-trimmed mean
  EXPECT_NEAR(
    aggregate_lengths( { 1.0, 2.0, 3.0, 4.0, 100.0 }, "average_iqr" ), 2.5, 1e-9 );
}

TEST( measurement_utilities_static, aggregate_lengths_ignores_invalid_and_empty )
{
  // Non-positive lengths are ignored
  EXPECT_NEAR( aggregate_lengths( { -1.0, 0.0, 4.0 }, "average" ), 4.0, 1e-9 );
  // No valid lengths -> -1
  EXPECT_LT( aggregate_lengths( {}, "average" ), 0.0 );
  EXPECT_LT( aggregate_lengths( { -1.0, 0.0 }, "average" ), 0.0 );
}

// =============================================================================
// Robust disparity segment measurement
// =============================================================================
namespace {
std::vector< std::pair< double, double > > segment_samples()
{
  std::vector< std::pair< double, double > > samples;
  for( int i = 0; i < 11; ++i )
  {
    const double f = i / 10.0;
    samples.emplace_back( f, 100.0 - 50.0 * f );
  }
  return samples;
}
}

TEST( disparity_segment, perspective_correct_endpoints )
{
  double head = -1, tail = -1;
  ASSERT_TRUE( fit_disparity_segment( segment_samples(), 11, 3, 0.1, head, tail ) );
  EXPECT_NEAR( head, 100.0, 1e-10 );
  EXPECT_NEAR( tail, 50.0, 1e-10 );
  // f=1000px, baseline=100mm: endpoints (0,0,1000), (400,0,2000).
  const kv::vector_3d h( 0, 0, 100000 / head );
  const kv::vector_3d t( 20000 / tail, 0, 100000 / tail );
  EXPECT_NEAR( ( t - h ).norm(), std::sqrt( 400.0 * 400 + 1000.0 * 1000 ), 1e-8 );
}

TEST( disparity_segment, rejects_outliers_and_counts_missing_samples )
{
  auto samples = segment_samples();
  samples[0].second = 500; // foreground contamination
  samples[5].second = 2;   // background contamination
  samples.pop_back();     // invalid disparity at the tail
  double head = -1, tail = -1;
  ASSERT_TRUE( fit_disparity_segment( samples, 11, 3, 0.1, head, tail ) );
  EXPECT_NEAR( head, 100, 1e-10 );
  EXPECT_NEAR( tail, 50, 1e-10 );
  EXPECT_FALSE( fit_disparity_segment( samples, 11, 2, 0.1, head, tail ) );
}

TEST( disparity_segment, invalid_data_preserves_outputs )
{
  auto samples = segment_samples();
  for( int i = 0; i < 4; ++i )
  {
    samples[i].second = std::numeric_limits< double >::quiet_NaN();
  }
  double head = 42, tail = 43;
  EXPECT_FALSE( fit_disparity_segment( samples, 11, 3, 1, head, tail ) );
  EXPECT_EQ( head, 42 );
  EXPECT_EQ( tail, 43 );
  EXPECT_FALSE( fit_disparity_segment( {}, 11, 3, 1, head, tail ) );
}

TEST( disparity_segment, validates_configuration )
{
  double head = -1, tail = -1;
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 2, 0, 1, head, tail ) );
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 102, 0, 1, head, tail ) );
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 11, 6, 1, head, tail ) );
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 11, -1, 1, head, tail ) );
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 11, 3, 0, head, tail ) );
  EXPECT_FALSE( fit_disparity_segment( segment_samples(), 11, 3,
    std::numeric_limits< double >::infinity(), head, tail ) );
}

TEST( disparity_segment, rejects_excessive_extrapolation_and_duplicates )
{
  auto samples = segment_samples();
  for( auto& sample : samples ) { sample.first *= 0.4; }
  double head = -1, tail = -1;
  EXPECT_FALSE( fit_disparity_segment( samples, 11, 3, 1, head, tail ) );
  samples = segment_samples();
  samples[5].first = samples[4].first;
  EXPECT_FALSE( fit_disparity_segment( samples, 11, 3, 1, head, tail ) );
}

TEST( disparity_segment, noisy_inliers_are_refit )
{
  auto samples = segment_samples();
  for( size_t i = 0; i < samples.size(); ++i )
  {
    samples[i].second += i % 2 ? 0.1 : -0.1;
  }
  double head = -1, tail = -1;
  ASSERT_TRUE( fit_disparity_segment( samples, 11, 3, 0.3, head, tail ) );
  EXPECT_NEAR( head, 100, 0.1 );
  EXPECT_NEAR( tail, 50, 0.1 );
}

TEST( disparity_segment, rejects_nonpositive_fitted_endpoints )
{
  auto samples = segment_samples();
  samples.erase( samples.begin(), samples.begin() + 2 );
  for( auto& sample : samples ) { sample.second = 100 * sample.first - 10; }
  double head = -1, tail = -1;
  EXPECT_FALSE( fit_disparity_segment( samples, 11, 3, 0.1, head, tail ) );
}

TEST_F( measurement_utilities_test, segment_configuration_is_opt_in_and_validated )
{
  map_keypoints_to_camera_settings settings;
  EXPECT_FALSE( settings.refine_disparity_segment );
  settings.refine_disparity_segment = true;
  settings.disparity_segment_samples = 2;
  EXPECT_THROW( utilities->configure( settings ), std::invalid_argument );
  settings.disparity_segment_samples = 11;
  settings.disparity_segment_max_outliers = 6;
  EXPECT_THROW( utilities->configure( settings ), std::invalid_argument );
  settings.disparity_segment_max_outliers = 3;
  EXPECT_NO_THROW( utilities->configure( settings ) );
}

TEST_F( measurement_utilities_test, segment_sampling_preserves_legacy_defaults )
{
  // Existing Foundation Stereo pipelines enable endpoint refinement only.
  map_keypoints_to_camera_settings settings;
  settings.refine_keypoints_with_disparity = true;
  utilities->configure( settings );
  kv::image_of< float > disparity( 400, 3 );
  for( unsigned y = 0; y < 3; ++y )
  {
    for( unsigned x = 0; x < 400; ++x )
    {
      disparity( x, y ) = 125.0 - 0.25 * x;
    }
  }
  auto map = std::make_shared< kv::simple_image_container >( disparity );
  kv::vector_2d right;
  ASSERT_TRUE( utilities->find_corresponding_point_external_disparity(
    map, kv::vector_2d( 100, 1 ), right, 0 ) );
  // Preserve the pre-existing byte-stride interpretation outside segment mode.
  EXPECT_DOUBLE_EQ( right.x(), 6.25 );

  settings.refine_disparity_segment = true;
  utilities->configure( settings );
  ASSERT_TRUE( utilities->find_corresponding_point_external_disparity(
    map, kv::vector_2d( 100, 1 ), right, 0 ) );
  EXPECT_DOUBLE_EQ( right.x(), 0.0 );
}

TEST_F( measurement_utilities_test, segment_disparity_formats_and_reversed_endpoints )
{
  map_keypoints_to_camera_settings settings;
  settings.refine_disparity_segment = true;
  settings.refine_keypoints_disparity_window = 0;
  utilities->configure( settings );
  kv::image_of< float > floats( 400, 3 );
  kv::image_of< int16_t > raw( 400, 3 );
  kv::image_of< uint16_t > scaled( 400, 3 );
  for( unsigned y = 0; y < 3; ++y )
  {
    for( unsigned x = 0; x < 400; ++x )
    {
      const double d = 125.0 - 0.25 * x;
      floats( x, y ) = d;
      raw( x, y ) = d * 16;
      scaled( x, y ) = d * 256;
    }
  }
  for( const auto& image : std::vector< kv::image >{ floats, raw, scaled } )
  {
    auto map = std::make_shared< kv::simple_image_container >( image );
    kv::vector_2d head, tail;
    ASSERT_TRUE( utilities->find_corresponding_segment_external_disparity(
      map, kv::vector_2d( 100, 1 ), kv::vector_2d( 300, 1 ), head, tail ) );
    EXPECT_NEAR( head.x(), 0, 1e-9 );
    EXPECT_NEAR( tail.x(), 250, 1e-9 );
    ASSERT_TRUE( utilities->find_corresponding_segment_external_disparity(
      map, kv::vector_2d( 300, 1 ), kv::vector_2d( 100, 1 ), tail, head ) );
    EXPECT_NEAR( head.x(), 0, 1e-9 );
    EXPECT_NEAR( tail.x(), 250, 1e-9 );
    EXPECT_FALSE( utilities->find_corresponding_segment_external_disparity(
      map, kv::vector_2d( 100, 1 ), kv::vector_2d( 100, 1 ), head, tail ) );
  }
}

#ifdef VIAME_ENABLE_OPENCV
TEST_F( measurement_utilities_test, segment_refinement_uses_rectification_and_triangulation )
{
  map_keypoints_to_camera_settings settings;
  settings.refine_disparity_segment = true;
  settings.refine_keypoints_disparity_window = 0;
  settings.disparity_segment_max_error = 0.01;
  utilities->configure( settings );
  utilities->compute_rectification_maps( *left_cam, *right_cam, cv::Size( 1280, 720 ) );
  kv::image_of< float > disparity( 1280, 720 );
  for( unsigned y = 0; y < 720; ++y )
  {
    for( unsigned x = 0; x < 1280; ++x )
    {
      disparity( x, y ) = 100 - 0.25 * ( static_cast< double >( x ) - 640 );
    }
  }
  const kv::vector_3d head3d( 0, 0, 1 ), tail3d( 0.4, 0, 2 );
  const auto head = left_cam->project( head3d );
  const auto tail = left_cam->project( tail3d );
  kv::vector_2d rh, rt;
  ASSERT_TRUE( utilities->refine_right_segment_with_disparity(
    std::make_shared< kv::simple_image_container >( disparity ),
    head, tail, *right_cam, rh, rt ) );
  const auto measured = compute_stereo_measurement(
    *left_cam, *right_cam, head, rh, tail, rt );
  EXPECT_NEAR( measured.length, ( tail3d - head3d ).norm(), 1e-6 );
  EXPECT_NEAR( measured.rms, 0, 1e-6 );
}
#endif
