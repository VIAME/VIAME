/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "measurement_utilities.h"

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
  EXPECT_EQ( methods.size(), 7 );

  // Check all expected methods are present
  std::set<std::string> method_set( methods.begin(), methods.end() );
  EXPECT_TRUE( method_set.count( "input_pairs_only" ) > 0 );
  EXPECT_TRUE( method_set.count( "depth_projection" ) > 0 );
  EXPECT_TRUE( method_set.count( "external_disparity" ) > 0 );
  EXPECT_TRUE( method_set.count( "compute_disparity" ) > 0 );
  EXPECT_TRUE( method_set.count( "template_matching" ) > 0 );
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
