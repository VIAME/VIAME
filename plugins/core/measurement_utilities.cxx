/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement utility functions implementation
 */

// Python.h must be included before any standard headers per Python C API docs
#ifdef VIAME_ENABLE_PYTHON
  #include <Python.h>
#endif

#include "measurement_utilities.h"

#include <vital/logger/logger.h>
#include <vital/util/string.h>

#include <arrows/mvg/triangulate.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <opencv2/imgcodecs.hpp>
  #include <opencv2/core/eigen.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>

namespace viame
{

namespace core
{

// =============================================================================
// DINOv3 Python C API helpers
// =============================================================================

#ifdef VIAME_ENABLE_OPENCV

namespace
{

static auto logger = kwiver::vital::get_logger( "viame.core.measurement_utilities" );

} // end anonymous namespace (logger)

#endif // VIAME_ENABLE_OPENCV

#if defined( VIAME_ENABLE_PYTHON ) && defined( VIAME_ENABLE_OPENCV )

namespace
{

/// RAII guard for acquiring and releasing the Python GIL
struct python_gil_guard
{
  PyGILState_STATE state;
  python_gil_guard() : state( PyGILState_Ensure() ) {}
  ~python_gil_guard() { PyGILState_Release( state ); }
};

/// RAII helper to decrement Python reference count on scope exit
struct py_decref_guard
{
  PyObject* obj;
  explicit py_decref_guard( PyObject* o ) : obj( o ) {}
  ~py_decref_guard() { Py_XDECREF( obj ); }
};

// Cached Python module reference (persists for process lifetime)
static PyObject* s_dino_module = nullptr;
static bool s_dino_init_attempted = false;
static bool s_dino_init_succeeded = false;

// Track which frame is currently loaded to avoid redundant set_images calls.
// Note: Cannot use raw data pointers because vital image containers may reuse
// the same memory buffer across frames, making pointer comparison unreliable.
static int64_t s_dino_cached_frame_id = -1;

/// Initialize the DINOv3 matcher module and model
bool dino_ensure_initialized(
  const std::string& model_name,
  double threshold,
  const std::string& weights_path )
{
  if( s_dino_init_succeeded )
  {
    return true;
  }

  if( s_dino_init_attempted )
  {
    return false;
  }

  s_dino_init_attempted = true;

  // Check if Python interpreter is running
  if( !Py_IsInitialized() )
  {
    LOG_WARN( logger, "DINO: Python interpreter not initialized" );
    return false;
  }

  python_gil_guard gil;

  // Import the matcher module
  s_dino_module = PyImport_ImportModule( "viame.pytorch.dino_matcher" );

  if( !s_dino_module )
  {
    LOG_WARN( logger, "DINO: Failed to import viame.pytorch.dino_matcher" );
    PyErr_Print();
    return false;
  }

  // Call init_matcher(model_name, device, threshold, weights_path)
  PyObject* result = PyObject_CallMethod(
    s_dino_module,
    "init_matcher",
    "ssds",
    model_name.c_str(),
    "cuda",
    threshold,
    weights_path.c_str() );

  if( !result )
  {
    LOG_WARN( logger, "DINO: init_matcher() failed" );
    PyErr_Print();
    return false;
  }

  Py_DECREF( result );
  s_dino_init_succeeded = true;

  LOG_INFO( logger, "DINO: Matcher initialized with model=" << model_name
    << " threshold=" << threshold );
  return true;
}

/// Call set_images_from_bytes to load a new stereo image pair
bool dino_set_images( const cv::Mat& left, const cv::Mat& right )
{
  if( !s_dino_module )
  {
    return false;
  }

  python_gil_guard gil;

  // Ensure images are contiguous
  cv::Mat left_cont = left.isContinuous() ? left : left.clone();
  cv::Mat right_cont = right.isContinuous() ? right : right.clone();

  // Create Python bytes objects wrapping the image data
  PyObject* left_bytes = PyBytes_FromStringAndSize(
    reinterpret_cast< const char* >( left_cont.data ),
    static_cast< Py_ssize_t >( left_cont.total() * left_cont.elemSize() ) );

  PyObject* right_bytes = PyBytes_FromStringAndSize(
    reinterpret_cast< const char* >( right_cont.data ),
    static_cast< Py_ssize_t >( right_cont.total() * right_cont.elemSize() ) );

  if( !left_bytes || !right_bytes )
  {
    Py_XDECREF( left_bytes );
    Py_XDECREF( right_bytes );
    PyErr_Print();
    return false;
  }

  // Call set_images_from_bytes(left_bytes, h, w, c, right_bytes, h, w, c)
  PyObject* result = PyObject_CallMethod(
    s_dino_module,
    "set_images_from_bytes",
    "OiiiOiii",
    left_bytes, left_cont.rows, left_cont.cols, left_cont.channels(),
    right_bytes, right_cont.rows, right_cont.cols, right_cont.channels() );

  Py_DECREF( left_bytes );
  Py_DECREF( right_bytes );

  if( !result )
  {
    LOG_WARN( logger, "DINO: set_images_from_bytes() failed" );
    PyErr_Print();
    return false;
  }

  bool ok = PyObject_IsTrue( result );
  Py_DECREF( result );

  return ok;
}

/// Match a single point along epipolar candidates
/// Returns (success, matched_x, matched_y, score)
struct dino_match_result
{
  bool success;
  double x, y, score;
};

dino_match_result dino_match_point(
  double src_x, double src_y,
  const std::vector< kv::vector_2d >& epipolar_points,
  double threshold )
{
  dino_match_result res = { false, 0.0, 0.0, 0.0 };

  if( !s_dino_module )
  {
    return res;
  }

  python_gil_guard gil;

  int n = static_cast< int >( epipolar_points.size() );

  // Build Python lists for epipolar x and y coordinates
  PyObject* xs = PyList_New( n );
  PyObject* ys = PyList_New( n );

  if( !xs || !ys )
  {
    Py_XDECREF( xs );
    Py_XDECREF( ys );
    return res;
  }

  for( int i = 0; i < n; ++i )
  {
    PyList_SET_ITEM( xs, i, PyFloat_FromDouble( epipolar_points[i].x() ) );
    PyList_SET_ITEM( ys, i, PyFloat_FromDouble( epipolar_points[i].y() ) );
  }

  // Call match_point(source_x, source_y, epipolar_xs, epipolar_ys, threshold)
  PyObject* result = PyObject_CallMethod(
    s_dino_module,
    "match_point",
    "ddOOd",
    src_x, src_y, xs, ys, threshold );

  Py_DECREF( xs );
  Py_DECREF( ys );

  if( !result )
  {
    PyErr_Print();
    return res;
  }

  // Parse result tuple: (success_bool, matched_x, matched_y, score)
  if( PyTuple_Check( result ) && PyTuple_Size( result ) == 4 )
  {
    res.success = PyObject_IsTrue( PyTuple_GET_ITEM( result, 0 ) );
    res.x = PyFloat_AsDouble( PyTuple_GET_ITEM( result, 1 ) );
    res.y = PyFloat_AsDouble( PyTuple_GET_ITEM( result, 2 ) );
    res.score = PyFloat_AsDouble( PyTuple_GET_ITEM( result, 3 ) );
  }

  Py_DECREF( result );
  return res;
}

/// Get top-K candidate indices from DINO cosine similarity ranking.
/// Returns indices into the original epipolar_points vector.
std::vector< int > dino_get_top_k_indices(
  double src_x, double src_y,
  const std::vector< kv::vector_2d >& epipolar_points,
  int k )
{
  std::vector< int > result;

  if( !s_dino_module || epipolar_points.empty() )
  {
    return result;
  }

  python_gil_guard gil;

  int n = static_cast< int >( epipolar_points.size() );

  PyObject* xs = PyList_New( n );
  PyObject* ys = PyList_New( n );

  if( !xs || !ys )
  {
    Py_XDECREF( xs );
    Py_XDECREF( ys );
    return result;
  }

  for( int i = 0; i < n; ++i )
  {
    PyList_SET_ITEM( xs, i, PyFloat_FromDouble( epipolar_points[i].x() ) );
    PyList_SET_ITEM( ys, i, PyFloat_FromDouble( epipolar_points[i].y() ) );
  }

  // Call get_top_k_indices(source_x, source_y, epipolar_xs, epipolar_ys, k)
  PyObject* py_result = PyObject_CallMethod(
    s_dino_module,
    "get_top_k_indices",
    "ddOOi",
    src_x, src_y, xs, ys, k );

  Py_DECREF( xs );
  Py_DECREF( ys );

  if( !py_result )
  {
    PyErr_Print();
    return result;
  }

  // Parse returned list of integer indices
  if( PyList_Check( py_result ) )
  {
    Py_ssize_t len = PyList_Size( py_result );
    result.reserve( static_cast< size_t >( len ) );

    for( Py_ssize_t i = 0; i < len; ++i )
    {
      PyObject* item = PyList_GET_ITEM( py_result, i );
      int idx = static_cast< int >( PyLong_AsLong( item ) );
      if( idx >= 0 && idx < n )
      {
        result.push_back( idx );
      }
    }
  }

  Py_DECREF( py_result );
  return result;
}

} // anonymous namespace

#endif // VIAME_ENABLE_PYTHON && VIAME_ENABLE_OPENCV

// =============================================================================
// map_keypoints_to_camera_settings implementation
// =============================================================================

// -----------------------------------------------------------------------------
map_keypoints_to_camera_settings
::map_keypoints_to_camera_settings()
  : matching_methods( "input_pairs_only,template_matching" )
  , default_depth( 5.0 )
  , template_size( 31 )
  , search_range( 128 )
  , template_matching_threshold( 0.2 )
  , template_matching_disparity( 0.0 )
  , use_disparity_hint( false )
  , use_multires_search( false )
  , multires_coarse_step( 4 )
  , use_census_transform( false )
  , epipolar_band_halfwidth( 0 )
  , epipolar_min_depth( 0.0 )
  , epipolar_max_depth( 0.0 )
  , epipolar_min_disparity( 0.0 )
  , epipolar_max_disparity( 0.0 )
  , epipolar_num_samples( 100 )
  , epipolar_descriptor_type( "ncc" )
  , use_distortion( true )
  , feature_search_radius( 50.0 )
  , ransac_inlier_scale( 3.0 )
  , min_ransac_inliers( 10 )
  , box_scale_factor( 1.10 )
  , box_min_aspect_ratio( 0.10 )
  , use_disparity_aware_feature_search( true )
  , feature_search_depth( 5.0 )
  , depth_consistency_max_ratio( 1.5 )
  , record_stereo_method( true )
  , debug_epipolar_directory( "" )
  , detection_pairing_method( "" )
  , detection_pairing_threshold( 0.1 )
  , dino_crop_max_area_ratio( 0.5 )
  , dino_model_name( "dinov2_vitb14" )
  , dino_threshold( 0.0 )
  , dino_weights_path( "" )
  , dino_top_k( 100 )
{
}

// -----------------------------------------------------------------------------
map_keypoints_to_camera_settings
::~map_keypoints_to_camera_settings()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
map_keypoints_to_camera_settings
::get_configuration() const
{
  kv::config_block_sptr config = kv::config_block::empty_config();

  config->set_value( "matching_methods", matching_methods,
    "Comma-separated list of methods to try (in order) for finding corresponding points "
    "in right camera for left-only tracks. Methods will be tried in the order specified "
    "until one succeeds. Valid options: "
    "'input_pairs_only' (use existing keypoints from right camera if available), "
    "'depth_projection' (uses default_depth to project points), "
    "'external_disparity' (uses externally provided disparity map), "
    "'compute_disparity' (uses stereo_disparity algorithm to compute disparity from rectified images), "
    "'template_matching' (rectifies images and searches along epipolar lines), "
    "'epipolar_template_matching' (matching along epipolar line on unrectified images, "
    "descriptor type controlled by epipolar_descriptor_type), "
    "'feature_descriptor' (uses vital feature detection/descriptor/matching), "
    "'ransac_feature' (feature matching with RANSAC-based fundamental matrix filtering). "
    "Example: 'input_pairs_only,compute_disparity,depth_projection'" );

  config->set_value( "default_depth", default_depth,
    "Default depth (in meters) to use when projecting left camera points to right camera "
    "for tracks that only exist in the left camera, when using the depth_projection option" );

  config->set_value( "template_size", template_size,
    "Template window size (in pixels) for template matching. Must be odd number." );

  config->set_value( "search_range", search_range,
    "Search range (in pixels) along epipolar line for template matching." );

  config->set_value( "template_matching_threshold", template_matching_threshold,
    "Minimum normalized correlation threshold for template matching (0.0 to 1.0). "
    "Higher values require better matches but may miss valid correspondences." );

  config->set_value( "template_matching_disparity", template_matching_disparity,
    "Expected disparity (in pixels) for centering the template matching search region. "
    "If set to 0 or negative, disparity is computed automatically from default_depth "
    "using the stereo camera parameters. Set this to override the automatic computation "
    "when the expected object depth differs from default_depth." );

  config->set_value( "use_disparity_hint", use_disparity_hint,
    "If true and SGBM disparity map is available, sample the disparity map near the "
    "query point to estimate initial disparity for template matching. This provides "
    "spatially-varying disparity estimates that can be more accurate than using a "
    "fixed default_depth for objects at varying distances." );

  config->set_value( "use_multires_search", use_multires_search,
    "If true, use multi-resolution search for template matching. First performs a "
    "coarse search with larger step size over the full search range, then refines "
    "around the best coarse match. This can significantly improve performance for "
    "large search ranges while maintaining accuracy." );

  config->set_value( "multires_coarse_step", multires_coarse_step,
    "Step size (in pixels) for the coarse search pass in multi-resolution template "
    "matching. Only used when use_multires_search is enabled. Larger values are "
    "faster but may miss optimal matches. Typical values are 2-4." );

  config->set_value( "use_census_transform", use_census_transform,
    "If true, apply census transform preprocessing before template matching. "
    "Census transform compares each pixel to its neighbors creating a binary pattern, "
    "which is highly robust to illumination changes and camera gain differences "
    "between stereo cameras." );

  config->set_value( "epipolar_band_halfwidth", epipolar_band_halfwidth,
    "Half-width of the epipolar band for template matching search (in pixels). "
    "Set to 0 for exact epipolar line search (single row, fastest). "
    "Set to 1-3 to allow small vertical deviation to handle imperfect rectification. "
    "The search will cover (2 * epipolar_band_halfwidth + 1) rows." );

  config->set_value( "epipolar_min_depth", epipolar_min_depth,
    "Minimum depth (in camera/calibration units) for epipolar template matching. "
    "Defines the near end of the depth range sampled along the camera ray. "
    "Default is 0 (off). Ignored when epipolar_min_disparity and "
    "epipolar_max_disparity are both > 0. Either disparity or depth parameters "
    "must be set for epipolar_template_matching to work." );

  config->set_value( "epipolar_max_depth", epipolar_max_depth,
    "Maximum depth (in camera/calibration units) for epipolar template matching. "
    "Defines the far end of the depth range sampled along the camera ray. "
    "Default is 0 (off). See epipolar_min_depth for details." );

  config->set_value( "epipolar_min_disparity", epipolar_min_disparity,
    "Minimum expected disparity in pixels for epipolar template matching "
    "(corresponds to the farthest objects). When both epipolar_min_disparity "
    "and epipolar_max_disparity are > 0, the depth range is computed "
    "automatically using: depth = focal_length * baseline / disparity. "
    "This is the recommended way to configure epipolar search range since "
    "disparity is unit-independent and can be estimated directly from the images." );

  config->set_value( "epipolar_max_disparity", epipolar_max_disparity,
    "Maximum expected disparity in pixels for epipolar template matching "
    "(corresponds to the nearest objects). See epipolar_min_disparity for details." );

  config->set_value( "epipolar_num_samples", epipolar_num_samples,
    "Number of sample points along the epipolar line for epipolar template matching. "
    "More samples give finer search resolution but take longer." );

  config->set_value( "epipolar_descriptor_type", epipolar_descriptor_type,
    "Descriptor type for epipolar template matching. "
    "'ncc' (default): normalized cross-correlation on grayscale patches (point-by-point). "
    "'ncc_strip': FFT-accelerated NCC on a strip covering the epipolar bounding box. "
    "Faster than point-by-point NCC for large candidate sets, no Python required. "
    "'dino': Two-stage DINO + NCC matching (requires Python). "
    "DINO features select the top-K semantically similar candidates, then NCC "
    "provides precise localization. This avoids NCC failures on repetitive textures "
    "while preserving sub-pixel accuracy. Set dino_top_k=0 for DINO-only mode." );

  config->set_value( "use_distortion", use_distortion,
    "Whether to use distortion coefficients from the calibration during rectification. "
    "If true, distortion coefficients from the calibration file are used. "
    "If false, zero distortion is assumed." );

  config->set_value( "feature_search_radius", feature_search_radius,
    "Maximum distance (in pixels) to search for feature matches around the expected location. "
    "Used for feature_descriptor and ransac_feature methods." );

  config->set_value( "ransac_inlier_scale", ransac_inlier_scale,
    "Inlier threshold for RANSAC fundamental matrix estimation. "
    "Points with reprojection error below this threshold are considered inliers." );

  config->set_value( "min_ransac_inliers", min_ransac_inliers,
    "Minimum number of inliers required for a valid RANSAC result." );

  config->set_value( "use_disparity_aware_feature_search", use_disparity_aware_feature_search,
    "If true, use depth projection to estimate the expected location of corresponding "
    "points in the right image when using feature_descriptor or ransac_feature methods. "
    "This helps account for stereo disparity when searching for feature matches, making "
    "the search more robust for objects at varying depths." );

  config->set_value( "feature_search_depth", feature_search_depth,
    "Depth (in meters) to use when estimating the expected location for disparity-aware "
    "feature search. If set to 0 or negative, uses the default_depth parameter instead. "
    "This allows using a different depth assumption for feature search than for the "
    "depth_projection matching method." );

  config->set_value( "box_scale_factor", box_scale_factor,
    "Scale factor to expand the bounding box around keypoints when creating "
    "new detections for the right image. A value of 1.10 means 10% expansion." );

  config->set_value( "box_min_aspect_ratio", box_min_aspect_ratio,
    "Minimum aspect ratio for bounding boxes (smaller dimension / larger dimension). "
    "Prevents very thin boxes when keypoints are nearly collinear. "
    "Set to 0 to disable. Default is 0.10 (10%)." );

  config->set_value( "depth_consistency_max_ratio", depth_consistency_max_ratio,
    "Maximum allowed depth ratio between head and tail keypoints when both are "
    "matched. If the deeper keypoint is more than this ratio times the shallower "
    "keypoint's depth, the deeper match is rejected (converted to a partial match "
    "with no length measurement). This catches false stereo matches where one "
    "keypoint incorrectly matched at the wrong depth. Set to 0 to disable. "
    "Default is 1.5 (50% depth difference allowed)." );

  config->set_value( "record_stereo_method", record_stereo_method,
    "If true, record the stereo measurement method used as an attribute on each "
    "output detection object. The attribute will be ':stereo_method=METHOD' "
    "where METHOD is one of: input_kps_used, template_matching, "
    "epipolar_template_matching, feature_descriptor, ransac_feature, "
    "depth_projection, external_disparity, or compute_disparity." );

  config->set_value( "debug_epipolar_directory", debug_epipolar_directory,
    "Directory to write debug images showing epipolar search lines overlaid on "
    "the source and target images. Each keypoint match attempt writes a side-by-side "
    "image with the source point marked on the left image and the bounded epipolar "
    "curve drawn on the right image, along with any matched point. "
    "Set to empty string (default) to disable debug output." );

  config->set_value( "detection_pairing_method", detection_pairing_method,
    "Method for pairing left/right detections that do not share the same track ID. "
    "Set to empty string (default) to disable detection pairing. "
    "Valid options: 'epipolar_iou' (project left bbox to right using depth, match by IOU), "
    "'keypoint_projection' (project left head/tail keypoints to right, match by pixel distance)." );

  config->set_value( "detection_pairing_threshold", detection_pairing_threshold,
    "Threshold for detection pairing. For 'epipolar_iou' method, this is the minimum IOU "
    "threshold (default 0.1). For 'keypoint_projection' method, this is the maximum average "
    "keypoint pixel distance (default 50.0)." );

  config->set_value( "dino_crop_max_area_ratio", dino_crop_max_area_ratio,
    "Maximum fraction of full image area that the union of left and right DINO "
    "crops may occupy. When all epipolar regions for a frame fit within this "
    "fraction, DINO runs on cropped subimages instead of the full resolution, "
    "proportionally reducing ViT inference cost. Set to 0 to disable cropping. "
    "Default 0.5 (50%)." );

  config->set_value( "dino_model_name", dino_model_name,
    "DINO backbone model name (used when epipolar_descriptor_type is 'dino'). "
    "Supports DINOv3 (e.g., 'dinov3_vits16') and DINOv2 (e.g., 'dinov2_vitb14'). "
    "If DINOv3 weights are unavailable, automatically falls back to DINOv2. "
    "DINOv2 options: 'dinov2_vits14' (small/fast), 'dinov2_vitb14' (base, recommended), "
    "'dinov2_vitl14' (large)." );

  config->set_value( "dino_threshold", dino_threshold,
    "Minimum cosine similarity threshold for DINO feature matching (0.0 to 1.0). "
    "With top-K + NCC mode (default, dino_top_k > 0), this is typically 0 since "
    "NCC provides the final selection. Only used for DINO-only mode (dino_top_k=0)." );

  config->set_value( "dino_weights_path", dino_weights_path,
    "Optional path to local DINO model weights file. If empty, weights are "
    "downloaded from the default URL on first use." );

  config->set_value( "dino_top_k", dino_top_k,
    "Number of top DINO candidates to pass to NCC for precise refinement. "
    "The two-stage approach (DINO top-K + NCC) combines DINO semantic robustness "
    "with NCC sub-pixel precision. Recommended value: 100. "
    "Set to 0 to use DINO-only matching without NCC refinement." );

  // Add nested algorithm configurations
  kv::algo::detect_features::get_nested_algo_configuration(
    "feature_detector", config, feature_detector );
  kv::algo::extract_descriptors::get_nested_algo_configuration(
    "descriptor_extractor", config, descriptor_extractor );
  kv::algo::match_features::get_nested_algo_configuration(
    "feature_matcher", config, feature_matcher );
  kv::algo::estimate_fundamental_matrix::get_nested_algo_configuration(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );
  kv::algo::compute_stereo_depth_map::get_nested_algo_configuration(
    "stereo_disparity", config, stereo_depth_map_algorithm );

  return config;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera_settings
::set_configuration( kv::config_block_sptr config )
{
  matching_methods = config->get_value< std::string >( "matching_methods", matching_methods );
  default_depth = config->get_value< double >( "default_depth", default_depth );
  template_size = config->get_value< int >( "template_size", template_size );
  search_range = config->get_value< int >( "search_range", search_range );
  template_matching_threshold = config->get_value< double >( "template_matching_threshold", template_matching_threshold );
  template_matching_disparity = config->get_value< double >( "template_matching_disparity", template_matching_disparity );
  use_disparity_hint = config->get_value< bool >( "use_disparity_hint", use_disparity_hint );
  use_multires_search = config->get_value< bool >( "use_multires_search", use_multires_search );
  multires_coarse_step = config->get_value< int >( "multires_coarse_step", multires_coarse_step );
  use_census_transform = config->get_value< bool >( "use_census_transform", use_census_transform );
  epipolar_band_halfwidth = config->get_value< int >( "epipolar_band_halfwidth", epipolar_band_halfwidth );
  epipolar_min_depth = config->get_value< double >( "epipolar_min_depth", epipolar_min_depth );
  epipolar_max_depth = config->get_value< double >( "epipolar_max_depth", epipolar_max_depth );
  epipolar_min_disparity = config->get_value< double >( "epipolar_min_disparity", epipolar_min_disparity );
  epipolar_max_disparity = config->get_value< double >( "epipolar_max_disparity", epipolar_max_disparity );
  epipolar_num_samples = config->get_value< int >( "epipolar_num_samples", epipolar_num_samples );
  epipolar_descriptor_type = config->get_value< std::string >( "epipolar_descriptor_type", epipolar_descriptor_type );
  use_distortion = config->get_value< bool >( "use_distortion", use_distortion );
  feature_search_radius = config->get_value< double >( "feature_search_radius", feature_search_radius );
  ransac_inlier_scale = config->get_value< double >( "ransac_inlier_scale", ransac_inlier_scale );
  min_ransac_inliers = config->get_value< int >( "min_ransac_inliers", min_ransac_inliers );
  box_scale_factor = config->get_value< double >( "box_scale_factor", box_scale_factor );
  box_min_aspect_ratio = config->get_value< double >( "box_min_aspect_ratio", box_min_aspect_ratio );
  use_disparity_aware_feature_search = config->get_value< bool >( "use_disparity_aware_feature_search", use_disparity_aware_feature_search );
  feature_search_depth = config->get_value< double >( "feature_search_depth", feature_search_depth );
  depth_consistency_max_ratio = config->get_value< double >( "depth_consistency_max_ratio", depth_consistency_max_ratio );
  record_stereo_method = config->get_value< bool >( "record_stereo_method", record_stereo_method );
  debug_epipolar_directory = config->get_value< std::string >( "debug_epipolar_directory", debug_epipolar_directory );
  detection_pairing_method = config->get_value< std::string >( "detection_pairing_method", detection_pairing_method );
  detection_pairing_threshold = config->get_value< double >( "detection_pairing_threshold", detection_pairing_threshold );
  dino_crop_max_area_ratio = config->get_value< double >( "dino_crop_max_area_ratio", dino_crop_max_area_ratio );
  dino_model_name = config->get_value< std::string >( "dino_model_name", dino_model_name );
  dino_threshold = config->get_value< double >( "dino_threshold", dino_threshold );
  dino_weights_path = config->get_value< std::string >( "dino_weights_path", dino_weights_path );
  dino_top_k = config->get_value< int >( "dino_top_k", dino_top_k );

  // Configure nested algorithms
  kv::algo::detect_features::set_nested_algo_configuration(
    "feature_detector", config, feature_detector );
  kv::algo::extract_descriptors::set_nested_algo_configuration(
    "descriptor_extractor", config, descriptor_extractor );
  kv::algo::match_features::set_nested_algo_configuration(
    "feature_matcher", config, feature_matcher );
  kv::algo::estimate_fundamental_matrix::set_nested_algo_configuration(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );
  kv::algo::compute_stereo_depth_map::set_nested_algo_configuration(
    "stereo_disparity", config, stereo_depth_map_algorithm );
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera_settings
::check_configuration( kv::config_block_sptr config ) const
{
  bool valid = true;

  // Check nested algorithms if present
  if( config->has_value( "feature_detector:type" ) &&
      config->get_value< std::string >( "feature_detector:type" ) != "" )
  {
    valid = kv::algo::detect_features::check_nested_algo_configuration(
      "feature_detector", config ) && valid;
  }
  if( config->has_value( "descriptor_extractor:type" ) &&
      config->get_value< std::string >( "descriptor_extractor:type" ) != "" )
  {
    valid = kv::algo::extract_descriptors::check_nested_algo_configuration(
      "descriptor_extractor", config ) && valid;
  }
  if( config->has_value( "feature_matcher:type" ) &&
      config->get_value< std::string >( "feature_matcher:type" ) != "" )
  {
    valid = kv::algo::match_features::check_nested_algo_configuration(
      "feature_matcher", config ) && valid;
  }
  if( config->has_value( "fundamental_matrix_estimator:type" ) &&
      config->get_value< std::string >( "fundamental_matrix_estimator:type" ) != "" )
  {
    valid = kv::algo::estimate_fundamental_matrix::check_nested_algo_configuration(
      "fundamental_matrix_estimator", config ) && valid;
  }
  if( config->has_value( "stereo_disparity:type" ) &&
      config->get_value< std::string >( "stereo_disparity:type" ) != "" )
  {
    valid = kv::algo::compute_stereo_depth_map::check_nested_algo_configuration(
      "stereo_disparity", config ) && valid;
  }

  return valid;
}

// -----------------------------------------------------------------------------
std::vector< std::string >
map_keypoints_to_camera_settings
::get_matching_methods() const
{
  return parse_matching_methods( matching_methods );
}

// -----------------------------------------------------------------------------
std::string
map_keypoints_to_camera_settings
::validate_matching_methods() const
{
  auto methods = get_matching_methods();

  if( methods.empty() )
  {
    return "No valid matching methods specified";
  }

  auto valid_methods = get_valid_methods();
  for( const auto& method : methods )
  {
    if( std::find( valid_methods.begin(), valid_methods.end(), method ) == valid_methods.end() )
    {
      return "Invalid matching method: " + method;
    }
  }

  return "";
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera_settings
::any_method_requires_images() const
{
  auto methods = get_matching_methods();
  for( const auto& method : methods )
  {
    if( method_requires_images( method ) )
    {
      return true;
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
std::vector< std::string >
map_keypoints_to_camera_settings
::check_feature_algorithm_warnings() const
{
  std::vector< std::string > warnings;

  auto methods = get_matching_methods();
  for( const auto& method : methods )
  {
    if( method == "feature_descriptor" || method == "ransac_feature" )
    {
      if( !feature_detector )
      {
        warnings.push_back( "Feature detector not configured; " + method + " method may not work" );
      }
      if( !descriptor_extractor )
      {
        warnings.push_back( "Descriptor extractor not configured; " + method + " method may not work" );
      }
      if( !feature_matcher )
      {
        warnings.push_back( "Feature matcher not configured; " + method + " method may not work" );
      }
      if( method == "ransac_feature" && !fundamental_matrix_estimator )
      {
        warnings.push_back( "Fundamental matrix estimator not configured; ransac_feature method may not work" );
      }
      break;  // Only need to check once
    }
  }

  return warnings;
}

// =============================================================================
// map_keypoints_to_camera implementation
// =============================================================================

// -----------------------------------------------------------------------------
map_keypoints_to_camera
::map_keypoints_to_camera()
  : m_default_depth( 5.0 )
  , m_template_size( 31 )
  , m_search_range( 128 )
  , m_template_matching_threshold( 0.2 )
  , m_template_matching_disparity( 0.0 )
  , m_use_disparity_hint( false )
  , m_use_multires_search( false )
  , m_multires_coarse_step( 4 )
  , m_use_census_transform( false )
  , m_epipolar_band_halfwidth( 0 )
  , m_epipolar_min_depth( 0.0 )
  , m_epipolar_max_depth( 0.0 )
  , m_epipolar_min_disparity( 0.0 )
  , m_epipolar_max_disparity( 0.0 )
  , m_epipolar_num_samples( 100 )
  , m_epipolar_descriptor_type( "ncc" )
  , m_use_distortion( true )
  , m_feature_search_radius( 50.0 )
  , m_ransac_inlier_scale( 3.0 )
  , m_min_ransac_inliers( 10 )
  , m_box_scale_factor( 1.10 )
  , m_box_min_aspect_ratio( 0.10 )
  , m_use_disparity_aware_feature_search( true )
  , m_feature_search_depth( 5.0 )
  , m_debug_epipolar_directory( "" )
  , m_debug_frame_counter( 0 )
  , m_dino_model_name( "dinov2_vitb14" )
  , m_dino_threshold( 0.0 )
  , m_dino_weights_path( "" )
  , m_dino_top_k( 100 )
  , m_dino_crop_max_area_ratio( 0.5 )
  , m_cached_frame_id( -1 )
#ifdef VIAME_ENABLE_OPENCV
  , m_dino_crop_active( false )
  , m_rectification_computed( false )
#endif
{
}

// -----------------------------------------------------------------------------
map_keypoints_to_camera
::~map_keypoints_to_camera()
{
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_default_depth( double depth )
{
  m_default_depth = depth;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_template_params( int template_size, int search_range,
                       double matching_threshold, double disparity,
                       bool use_sgbm_hint, bool use_multires,
                       int multires_step, bool use_census,
                       int epipolar_band )
{
  m_template_size = template_size;
  m_search_range = search_range;
  m_template_matching_threshold = matching_threshold;
  m_template_matching_disparity = disparity;
  m_use_disparity_hint = use_sgbm_hint;
  m_use_multires_search = use_multires;
  m_multires_coarse_step = multires_step;
  m_use_census_transform = use_census;
  m_epipolar_band_halfwidth = epipolar_band;

  // Ensure template size is odd
  if( m_template_size % 2 == 0 )
  {
    m_template_size++;
  }

  // Ensure coarse step is at least 2 for multi-res to be useful
  if( m_multires_coarse_step < 2 )
  {
    m_multires_coarse_step = 2;
  }

  // Clamp epipolar band to reasonable range
  if( m_epipolar_band_halfwidth < 0 )
  {
    m_epipolar_band_halfwidth = 0;
  }
  else if( m_epipolar_band_halfwidth > 10 )
  {
    m_epipolar_band_halfwidth = 10;
  }
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_epipolar_params( double min_depth, double max_depth, int num_samples )
{
  m_epipolar_min_depth = min_depth;
  m_epipolar_max_depth = max_depth;
  m_epipolar_num_samples = num_samples;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_use_distortion( bool use_distortion )
{
  m_use_distortion = use_distortion;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_feature_params( double search_radius, double ransac_inlier_scale,
                      int min_ransac_inliers,
                      bool use_disparity_aware_search,
                      double feature_search_depth )
{
  m_feature_search_radius = search_radius;
  m_ransac_inlier_scale = ransac_inlier_scale;
  m_min_ransac_inliers = min_ransac_inliers;
  m_use_disparity_aware_feature_search = use_disparity_aware_search;
  m_feature_search_depth = feature_search_depth;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_box_scale_factor( double scale_factor )
{
  m_box_scale_factor = scale_factor;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_feature_algorithms(
  kv::algo::detect_features_sptr detector,
  kv::algo::extract_descriptors_sptr extractor,
  kv::algo::match_features_sptr matcher,
  kv::algo::estimate_fundamental_matrix_sptr fundamental_estimator )
{
  m_feature_detector = detector;
  m_descriptor_extractor = extractor;
  m_feature_matcher = matcher;
  m_fundamental_matrix_estimator = fundamental_estimator;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::configure( const map_keypoints_to_camera_settings& settings )
{
  set_default_depth( settings.default_depth );
  set_template_params( settings.template_size, settings.search_range,
                       settings.template_matching_threshold,
                       settings.template_matching_disparity,
                       settings.use_disparity_hint,
                       settings.use_multires_search,
                       settings.multires_coarse_step,
                       settings.use_census_transform,
                       settings.epipolar_band_halfwidth );
  set_epipolar_params( settings.epipolar_min_depth, settings.epipolar_max_depth,
                       settings.epipolar_num_samples );
  m_epipolar_min_disparity = settings.epipolar_min_disparity;
  m_epipolar_max_disparity = settings.epipolar_max_disparity;
  m_epipolar_descriptor_type = settings.epipolar_descriptor_type;
  set_use_distortion( settings.use_distortion );
  set_feature_params( settings.feature_search_radius, settings.ransac_inlier_scale,
                      settings.min_ransac_inliers,
                      settings.use_disparity_aware_feature_search,
                      settings.feature_search_depth );
  set_box_scale_factor( settings.box_scale_factor );
  m_box_min_aspect_ratio = settings.box_min_aspect_ratio;
  set_feature_algorithms( settings.feature_detector, settings.descriptor_extractor,
                          settings.feature_matcher, settings.fundamental_matrix_estimator );

  m_debug_epipolar_directory = settings.debug_epipolar_directory;

  m_dino_model_name = settings.dino_model_name;
  m_dino_threshold = settings.dino_threshold;
  m_dino_weights_path = settings.dino_weights_path;
  m_dino_top_k = settings.dino_top_k;
  m_dino_crop_max_area_ratio = settings.dino_crop_max_area_ratio;

  // Set the stereo depth map algorithm for compute_disparity method
  m_stereo_depth_map_algorithm = settings.stereo_depth_map_algorithm;
}

// -----------------------------------------------------------------------------
std::string
map_keypoints_to_camera
::epipolar_descriptor_type() const
{
  return m_epipolar_descriptor_type;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::clear_dino_crop_info()
{
#ifdef VIAME_ENABLE_OPENCV
  m_dino_crop_active = false;
  m_dino_left_cropped = cv::Mat();
  m_dino_right_cropped = cv::Mat();
#endif
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::precompute_dino_crops(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const std::vector< kv::vector_2d >& all_left_heads,
  const std::vector< kv::vector_2d >& all_left_tails,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image )
{
#ifdef VIAME_ENABLE_OPENCV
  m_dino_crop_active = false;

  if( m_dino_crop_max_area_ratio <= 0.0 || !left_image || !right_image )
  {
    return;
  }

  int img_w = static_cast< int >( left_image->width() );
  int img_h = static_cast< int >( left_image->height() );
  double full_area = static_cast< double >( img_w * img_h );

  if( full_area <= 0 )
  {
    return;
  }

  // Determine effective depth range (same logic as find_stereo_correspondence)
  double eff_min_depth = m_epipolar_min_depth;
  double eff_max_depth = m_epipolar_max_depth;

  if( m_epipolar_min_disparity > 0.0 && m_epipolar_max_disparity > 0.0 )
  {
    double fx = left_cam.get_intrinsics()->focal_length();
    double baseline = ( left_cam.center() - right_cam.center() ).norm();

    eff_min_depth = fx * baseline / m_epipolar_max_disparity;
    eff_max_depth = fx * baseline / m_epipolar_min_disparity;
  }

  if( eff_min_depth <= 0.0 || eff_max_depth <= 0.0 )
  {
    return;
  }

  // Accumulate bounding boxes for all source keypoints (left crop)
  // and all epipolar points (right crop)
  double left_min_x = std::numeric_limits< double >::max();
  double left_min_y = std::numeric_limits< double >::max();
  double left_max_x = std::numeric_limits< double >::lowest();
  double left_max_y = std::numeric_limits< double >::lowest();

  double right_min_x = std::numeric_limits< double >::max();
  double right_min_y = std::numeric_limits< double >::max();
  double right_max_x = std::numeric_limits< double >::lowest();
  double right_max_y = std::numeric_limits< double >::lowest();

  // Gather all source keypoints
  std::vector< kv::vector_2d > all_kps;
  all_kps.reserve( all_left_heads.size() + all_left_tails.size() );
  all_kps.insert( all_kps.end(), all_left_heads.begin(), all_left_heads.end() );
  all_kps.insert( all_kps.end(), all_left_tails.begin(), all_left_tails.end() );

  for( const auto& kp : all_kps )
  {
    left_min_x = std::min( left_min_x, kp.x() );
    left_min_y = std::min( left_min_y, kp.y() );
    left_max_x = std::max( left_max_x, kp.x() );
    left_max_y = std::max( left_max_y, kp.y() );

    // Compute epipolar points for this keypoint
    auto epi_pts = compute_epipolar_points(
      left_cam, right_cam, kp, eff_min_depth, eff_max_depth, m_epipolar_num_samples );

    for( const auto& ep : epi_pts )
    {
      right_min_x = std::min( right_min_x, ep.x() );
      right_min_y = std::min( right_min_y, ep.y() );
      right_max_x = std::max( right_max_x, ep.x() );
      right_max_y = std::max( right_max_y, ep.y() );
    }
  }

  // Check we got valid bounds
  if( left_min_x > left_max_x || right_min_x > right_max_x )
  {
    return;
  }

  // Expand by template_size/2 + 16px padding, then align to patch_size 14
  int pad = m_template_size / 2 + 16;
  int patch_size = 14;

  auto align_crop = [&]( double mn_x, double mn_y, double mx_x, double mx_y,
                         int iw, int ih ) -> cv::Rect
  {
    int x0 = static_cast< int >( std::floor( mn_x ) ) - pad;
    int y0 = static_cast< int >( std::floor( mn_y ) ) - pad;
    int x1 = static_cast< int >( std::ceil( mx_x ) ) + pad;
    int y1 = static_cast< int >( std::ceil( mx_y ) ) + pad;

    // Clamp to image bounds
    x0 = std::max( 0, x0 );
    y0 = std::max( 0, y0 );
    x1 = std::min( iw, x1 );
    y1 = std::min( ih, y1 );

    // Align dimensions to patch_size (round up)
    int w = x1 - x0;
    int h = y1 - y0;
    int rem_w = w % patch_size;
    int rem_h = h % patch_size;
    if( rem_w != 0 )
    {
      int extra = patch_size - rem_w;
      // Try to expand right, then left
      if( x1 + extra <= iw ) x1 += extra;
      else x0 = std::max( 0, x0 - extra );
    }
    if( rem_h != 0 )
    {
      int extra = patch_size - rem_h;
      if( y1 + extra <= ih ) y1 += extra;
      else y0 = std::max( 0, y0 - extra );
    }

    return cv::Rect( x0, y0, x1 - x0, y1 - y0 );
  };

  cv::Rect left_crop = align_crop( left_min_x, left_min_y, left_max_x, left_max_y, img_w, img_h );

  int right_img_w = static_cast< int >( right_image->width() );
  int right_img_h = static_cast< int >( right_image->height() );
  cv::Rect right_crop = align_crop( right_min_x, right_min_y, right_max_x, right_max_y,
                                     right_img_w, right_img_h );

  // Check area ratio
  double left_area = static_cast< double >( left_crop.width * left_crop.height );
  double right_area = static_cast< double >( right_crop.width * right_crop.height );
  double right_full_area = static_cast< double >( right_img_w * right_img_h );
  double avg_ratio = ( left_area / full_area + right_area / right_full_area ) / 2.0;

  if( avg_ratio >= m_dino_crop_max_area_ratio )
  {
    // Crops are too large — run DINO on full images instead
    return;
  }

  // Extract cropped images
  cv::Mat left_bgr = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat right_bgr = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  m_dino_left_crop = left_crop;
  m_dino_right_crop = right_crop;
  m_dino_left_cropped = left_bgr( left_crop ).clone();
  m_dino_right_cropped = right_bgr( right_crop ).clone();
  m_dino_crop_active = true;

  LOG_INFO( logger, "DINO crop: left " << left_crop.width << "x" << left_crop.height
    << " right " << right_crop.width << "x" << right_crop.height
    << " (avg ratio " << avg_ratio << ")" );
#endif
}

// -----------------------------------------------------------------------------
kv::vector_2d
map_keypoints_to_camera
::project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point ) const
{
  return viame::core::project_left_to_right( left_cam, right_cam, left_point, m_default_depth );
}

// -----------------------------------------------------------------------------
kv::bounding_box_d
map_keypoints_to_camera
::compute_bbox_from_keypoints(
  const kv::vector_2d& head_point,
  const kv::vector_2d& tail_point ) const
{
  return viame::core::compute_bbox_from_keypoints(
    head_point, tail_point, m_box_scale_factor, m_box_min_aspect_ratio );
}

// -----------------------------------------------------------------------------
void
add_measurement_attributes(
  kv::detected_object_sptr det,
  const stereo_measurement_result& measurement )
{
  det->set_length( measurement.length );
  det->add_note( ":midpoint_x=" + std::to_string( measurement.x ) );
  det->add_note( ":midpoint_y=" + std::to_string( measurement.y ) );
  det->add_note( ":midpoint_z=" + std::to_string( measurement.z ) );
  det->add_note( ":midpoint_range=" + std::to_string( measurement.range ) );
  det->add_note( ":stereo_rms=" + std::to_string( measurement.rms ) );
}

// -----------------------------------------------------------------------------
kv::vector_2d
project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  double depth )
{
  // Unproject the left camera point to normalized image coordinates
  const auto left_intrinsics = left_cam.get_intrinsics();
  const kv::vector_2d normalized_pt = left_intrinsics->unmap( left_point );

  // Convert to homogeneous coordinates and add depth
  kv::vector_3d ray_direction( normalized_pt.x(), normalized_pt.y(), 1.0 );
  ray_direction.normalize();

  // Compute 3D point at specified depth in left camera coordinates
  kv::vector_3d point_3d_left_cam = ray_direction * depth;

  // Transform to world coordinates
  const auto& left_rotation = left_cam.rotation();
  const auto& left_center = left_cam.center();
  kv::vector_3d point_3d_world = left_rotation.inverse() * point_3d_left_cam + left_center;

  // Transform to right camera coordinates
  const auto& right_rotation = right_cam.rotation();
  const auto& right_center = right_cam.center();
  kv::vector_3d point_3d_right_cam = right_rotation * ( point_3d_world - right_center );

  // Project to right camera image
  const auto right_intrinsics = right_cam.get_intrinsics();
  kv::vector_2d normalized_right( point_3d_right_cam.x() / point_3d_right_cam.z(),
                                   point_3d_right_cam.y() / point_3d_right_cam.z() );
  return right_intrinsics->map( normalized_right );
}

// -----------------------------------------------------------------------------
kv::vector_3d
triangulate_point(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point )
{
  Eigen::Matrix<double, 2, 1> left_pt( left_point.x(), left_point.y() );
  Eigen::Matrix<double, 2, 1> right_pt( right_point.x(), right_point.y() );

  auto point_3d = kwiver::arrows::mvg::triangulate_fast_two_view(
    left_cam, right_cam, left_pt, right_pt );

  return kv::vector_3d( point_3d.x(), point_3d.y(), point_3d.z() );
}

// -----------------------------------------------------------------------------
double
compute_stereo_length(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail )
{
  kv::vector_3d head_3d = triangulate_point( left_cam, right_cam, left_head, right_head );
  kv::vector_3d tail_3d = triangulate_point( left_cam, right_cam, left_tail, right_tail );

  return ( tail_3d - head_3d ).norm();
}

// -----------------------------------------------------------------------------
stereo_measurement_result
compute_stereo_measurement(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail )
{
  stereo_measurement_result result;

  // Triangulate head and tail points
  kv::vector_3d head_3d = triangulate_point( left_cam, right_cam, left_head, right_head );
  kv::vector_3d tail_3d = triangulate_point( left_cam, right_cam, left_tail, right_tail );

  // Compute length
  result.length = ( tail_3d - head_3d ).norm();

  // Compute midpoint (real-world 3D location)
  kv::vector_3d midpoint_3d = ( head_3d + tail_3d ) / 2.0;
  result.x = midpoint_3d.x();
  result.y = midpoint_3d.y();
  result.z = midpoint_3d.z();

  // Compute range (distance from midpoint to left camera center)
  const kv::vector_3d& left_center = left_cam.center();
  result.range = ( midpoint_3d - left_center ).norm();

  // Compute RMS reprojection error
  // Project the 3D points back to both cameras and measure error
  auto compute_reprojection_error = [&]( const kv::vector_3d& pt_3d,
                                          const kv::vector_2d& left_pt,
                                          const kv::vector_2d& right_pt ) -> double
  {
    // Project to left camera
    kv::vector_2d left_reproj = left_cam.project( pt_3d );
    double left_err_sq = ( left_reproj - left_pt ).squaredNorm();

    // Project to right camera
    kv::vector_2d right_reproj = right_cam.project( pt_3d );
    double right_err_sq = ( right_reproj - right_pt ).squaredNorm();

    return left_err_sq + right_err_sq;
  };

  double head_err_sq = compute_reprojection_error( head_3d, left_head, right_head );
  double tail_err_sq = compute_reprojection_error( tail_3d, left_tail, right_tail );

  // RMS = sqrt( sum of squared errors / number of measurements )
  // 4 measurements total: left_head, right_head, left_tail, right_tail
  result.rms = std::sqrt( ( head_err_sq + tail_err_sq ) / 4.0 );

  result.valid = true;
  return result;
}

// -----------------------------------------------------------------------------
kv::bounding_box_d
compute_bbox_from_keypoints(
  const kv::vector_2d& head_point,
  const kv::vector_2d& tail_point,
  double box_scale_factor,
  double min_aspect_ratio )
{
  // Compute bounding box around the keypoints
  double min_x = std::min( head_point.x(), tail_point.x() );
  double max_x = std::max( head_point.x(), tail_point.x() );
  double min_y = std::min( head_point.y(), tail_point.y() );
  double max_y = std::max( head_point.y(), tail_point.y() );

  // Compute center and dimensions
  double center_x = ( min_x + max_x ) / 2.0;
  double center_y = ( min_y + max_y ) / 2.0;
  double width = max_x - min_x;
  double height = max_y - min_y;

  // Apply scale factor
  double scaled_width = width * box_scale_factor;
  double scaled_height = height * box_scale_factor;

  // Enforce minimum aspect ratio (smaller dimension >= min_aspect_ratio * larger dimension)
  if( min_aspect_ratio > 0.0 )
  {
    if( scaled_width > scaled_height )
    {
      double min_height = scaled_width * min_aspect_ratio;
      if( scaled_height < min_height )
      {
        scaled_height = min_height;
      }
    }
    else
    {
      double min_width = scaled_height * min_aspect_ratio;
      if( scaled_width < min_width )
      {
        scaled_width = min_width;
      }
    }
  }

  // Compute new bounding box coordinates
  double new_min_x = center_x - scaled_width / 2.0;
  double new_max_x = center_x + scaled_width / 2.0;
  double new_min_y = center_y - scaled_height / 2.0;
  double new_max_y = center_y + scaled_height / 2.0;

  return kv::bounding_box_d( new_min_x, new_min_y, new_max_x, new_max_y );
}

// -----------------------------------------------------------------------------
std::vector< kv::vector_2d >
compute_epipolar_points(
  const kv::simple_camera_perspective& source_cam,
  const kv::simple_camera_perspective& target_cam,
  const kv::vector_2d& source_point,
  double min_depth, double max_depth, int num_samples )
{
  std::vector< kv::vector_2d > points;
  points.reserve( num_samples );

  // Unproject source point to normalized image coordinates
  const auto source_intrinsics = source_cam.get_intrinsics();
  const kv::vector_2d normalized_pt = source_intrinsics->unmap( source_point );

  // Ray direction in source camera coordinates
  kv::vector_3d ray_direction( normalized_pt.x(), normalized_pt.y(), 1.0 );
  ray_direction.normalize();

  // Source camera pose
  const auto& source_rotation = source_cam.rotation();
  const auto& source_center = source_cam.center();

  // Target camera pose and intrinsics
  const auto& target_rotation = target_cam.rotation();
  const auto& target_center = target_cam.center();
  const auto target_intrinsics = target_cam.get_intrinsics();

  double depth_step = ( num_samples > 1 )
    ? ( max_depth - min_depth ) / ( num_samples - 1 ) : 0.0;

  // Track last emitted integer pixel to skip duplicate samples that
  // project to the same pixel (common when depth sampling is dense
  // relative to the epipolar line length in the target image).
  int prev_px = std::numeric_limits< int >::min();
  int prev_py = std::numeric_limits< int >::min();

  for( int i = 0; i < num_samples; ++i )
  {
    double depth = min_depth + i * depth_step;

    // 3D point along ray in source camera coordinates
    kv::vector_3d point_3d_cam = ray_direction * depth;

    // Transform to world coordinates
    kv::vector_3d point_3d_world = source_rotation.inverse() * point_3d_cam + source_center;

    // Transform to target camera coordinates
    kv::vector_3d point_3d_target = target_rotation * ( point_3d_world - target_center );

    // Skip points behind target camera
    if( point_3d_target.z() <= 0.0 )
    {
      continue;
    }

    // Project to target image
    kv::vector_2d normalized_target( point_3d_target.x() / point_3d_target.z(),
                                      point_3d_target.y() / point_3d_target.z() );
    kv::vector_2d projected = target_intrinsics->map( normalized_target );

    // Skip if this rounds to the same pixel as the previous point
    int px = static_cast< int >( projected.x() + 0.5 );
    int py = static_cast< int >( projected.y() + 0.5 );

    if( px == prev_px && py == prev_py )
    {
      continue;
    }

    prev_px = px;
    prev_py = py;
    points.push_back( projected );
  }

  return points;
}

// =============================================================================
// Free-standing utility function implementations
// =============================================================================

// -----------------------------------------------------------------------------
double
compute_iou(
  const kv::bounding_box_d& bbox1,
  const kv::bounding_box_d& bbox2 )
{
  if( !bbox1.is_valid() || !bbox2.is_valid() )
  {
    return 0.0;
  }

  // Compute intersection
  double x1 = std::max( bbox1.min_x(), bbox2.min_x() );
  double y1 = std::max( bbox1.min_y(), bbox2.min_y() );
  double x2 = std::min( bbox1.max_x(), bbox2.max_x() );
  double y2 = std::min( bbox1.max_y(), bbox2.max_y() );

  double intersection_width = std::max( 0.0, x2 - x1 );
  double intersection_height = std::max( 0.0, y2 - y1 );
  double intersection_area = intersection_width * intersection_height;

  if( intersection_area <= 0.0 )
  {
    return 0.0;
  }

  // Compute union
  double area1 = bbox1.width() * bbox1.height();
  double area2 = bbox2.width() * bbox2.height();
  double union_area = area1 + area2 - intersection_area;

  if( union_area <= 0.0 )
  {
    return 0.0;
  }

  return intersection_area / union_area;
}

// -----------------------------------------------------------------------------
std::string
get_detection_class_label( const kv::detected_object_sptr& det )
{
  if( !det )
  {
    return "";
  }

  auto det_type = det->type();
  if( !det_type )
  {
    return "";
  }

  std::string most_likely;
  det_type->get_most_likely( most_likely );
  return most_likely;
}

// -----------------------------------------------------------------------------
std::vector< std::pair< int, int > >
greedy_assignment(
  const std::vector< std::vector< double > >& cost_matrix,
  int n_rows, int n_cols )
{
  std::vector< std::pair< int, int > > assignment;
  std::vector< bool > row_used( n_rows, false );
  std::vector< bool > col_used( n_cols, false );

  // Collect all valid costs with their indices
  std::vector< std::tuple< double, int, int > > costs;
  for( int i = 0; i < n_rows; ++i )
  {
    for( int j = 0; j < n_cols; ++j )
    {
      double cost = cost_matrix[i][j];
      if( std::isfinite( cost ) && cost < 1e9 )
      {
        costs.push_back( std::make_tuple( cost, i, j ) );
      }
    }
  }

  // Sort by cost (ascending - lower is better)
  std::sort( costs.begin(), costs.end() );

  // Greedily assign
  for( const auto& entry : costs )
  {
    int i = std::get< 1 >( entry );
    int j = std::get< 2 >( entry );

    if( !row_used[i] && !col_used[j] )
    {
      assignment.push_back( std::make_pair( i, j ) );
      row_used[i] = true;
      col_used[j] = true;
    }
  }

  return assignment;
}

// -----------------------------------------------------------------------------
bool
find_furthest_apart_points(
  const std::vector< stereo_feature_correspondence >& correspondences,
  kv::vector_2d& left_head, kv::vector_2d& left_tail,
  kv::vector_2d& right_head, kv::vector_2d& right_tail )
{
  if( correspondences.size() < 2 )
  {
    return false;
  }

  // Find the two points in the left image that are furthest apart
  double max_dist_sq = 0.0;
  size_t best_i = 0, best_j = 1;

  for( size_t i = 0; i < correspondences.size(); ++i )
  {
    for( size_t j = i + 1; j < correspondences.size(); ++j )
    {
      double dist_sq = ( correspondences[i].left_point -
                         correspondences[j].left_point ).squaredNorm();
      if( dist_sq > max_dist_sq )
      {
        max_dist_sq = dist_sq;
        best_i = i;
        best_j = j;
      }
    }
  }

  // Assign head and tail based on which point is more to the left (lower x)
  // This provides a consistent ordering
  if( correspondences[best_i].left_point.x() < correspondences[best_j].left_point.x() )
  {
    left_head = correspondences[best_i].left_point;
    left_tail = correspondences[best_j].left_point;
    right_head = correspondences[best_i].right_point;
    right_tail = correspondences[best_j].right_point;
  }
  else
  {
    left_head = correspondences[best_j].left_point;
    left_tail = correspondences[best_i].left_point;
    right_head = correspondences[best_j].right_point;
    right_tail = correspondences[best_i].right_point;
  }

  return true;
}

// -----------------------------------------------------------------------------
map_keypoints_to_camera::stereo_correspondence_result
map_keypoints_to_camera
::find_stereo_correspondence(
  const std::vector< std::string >& methods,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d* right_head_input,
  const kv::vector_2d* right_tail_input,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  const kv::image_container_sptr& external_disparity )
{
  stereo_correspondence_result result;
  result.success = false;
  result.head_found = false;
  result.tail_found = false;
  result.left_head = left_head;
  result.left_tail = left_tail;

  bool head_found = false;
  bool tail_found = false;

#ifdef VIAME_ENABLE_OPENCV
  // Prepare stereo images if needed
  bool has_images = ( left_image && right_image );
  if( has_images )
  {
    m_cached_stereo_images = prepare_stereo_images(
      methods, left_cam, right_cam, left_image, right_image );
  }
#endif

  for( const auto& method : methods )
  {
    if( head_found && tail_found )
    {
      break;
    }

    if( method == "input_pairs_only" )
    {
      if( right_head_input && right_tail_input )
      {
        result.right_head = *right_head_input;
        result.right_tail = *right_tail_input;
        head_found = true;
        tail_found = true;
        result.method_used = "input_pairs_only";
      }
    }
    else if( method == "depth_projection" )
    {
      result.right_head = project_left_to_right( left_cam, right_cam, result.left_head );
      result.right_tail = project_left_to_right( left_cam, right_cam, result.left_tail );
      head_found = true;
      tail_found = true;
      result.method_used = "depth_projection";
    }
    else if( method == "external_disparity" && external_disparity )
    {
      head_found = find_corresponding_point_external_disparity(
        external_disparity, result.left_head, result.right_head );
      tail_found = find_corresponding_point_external_disparity(
        external_disparity, result.left_tail, result.right_tail );

      if( head_found || tail_found )
      {
        result.method_used = "external_disparity";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
#ifdef VIAME_ENABLE_OPENCV
    else if( method == "compute_disparity" && m_stereo_depth_map_algorithm &&
             m_cached_stereo_images.rectified_available )
    {
      // Compute disparity map using the configured algorithm if not cached for this frame
      if( !m_cached_compute_disparity )
      {
        // Convert rectified cv::Mat images to ImageContainers using OCV wrapper
        kv::image_container_sptr left_rect_container =
          std::make_shared< kwiver::arrows::ocv::image_container >(
            m_cached_stereo_images.left_rectified,
            kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );
        kv::image_container_sptr right_rect_container =
          std::make_shared< kwiver::arrows::ocv::image_container >(
            m_cached_stereo_images.right_rectified,
            kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );

        // Compute disparity using the configured algorithm
        m_cached_compute_disparity = m_stereo_depth_map_algorithm->compute(
          left_rect_container, right_rect_container );
      }

      if( m_cached_compute_disparity )
      {
        // Use the computed disparity to find correspondences
        // Note: The disparity is in rectified image space, so we need to
        // rectify points, find correspondences, then unrectify
        kv::vector_2d left_head_rect = rectify_point( result.left_head, false );
        kv::vector_2d left_tail_rect = rectify_point( result.left_tail, false );

        kv::vector_2d right_head_rect, right_tail_rect;

        head_found = find_corresponding_point_external_disparity(
          m_cached_compute_disparity, left_head_rect, right_head_rect, 7 );
        tail_found = find_corresponding_point_external_disparity(
          m_cached_compute_disparity, left_tail_rect, right_tail_rect, 7 );

        if( head_found || tail_found )
        {
          // Unrectify the found right image points
          if( head_found )
            result.right_head = unrectify_point( right_head_rect, true, right_cam );
          if( tail_found )
            result.right_tail = unrectify_point( right_tail_rect, true, right_cam );
          result.method_used = "compute_disparity";
        }
        else
        {
          head_found = false;
          tail_found = false;
        }
      }
    }
    else if( method == "template_matching" && m_cached_stereo_images.rectified_available )
    {
      kv::vector_2d left_head_rect = rectify_point( result.left_head, false );
      kv::vector_2d left_tail_rect = rectify_point( result.left_tail, false );

      // Pass disparity map for SGBM hint if available
      const cv::Mat& disp_hint = m_cached_stereo_images.disparity_available ?
        m_cached_stereo_images.disparity_map : cv::Mat();

      kv::vector_2d right_head_rect, right_tail_rect;
      head_found = find_corresponding_point_template_matching(
        m_cached_stereo_images.left_rectified, m_cached_stereo_images.right_rectified,
        left_head_rect, right_head_rect, disp_hint );
      tail_found = find_corresponding_point_template_matching(
        m_cached_stereo_images.left_rectified, m_cached_stereo_images.right_rectified,
        left_tail_rect, right_tail_rect, disp_hint );

      if( head_found || tail_found )
      {
        if( head_found )
          result.right_head = unrectify_point( right_head_rect, true, right_cam );
        if( tail_found )
          result.right_tail = unrectify_point( right_tail_rect, true, right_cam );
        result.method_used = "template_matching";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
    else if( method == "epipolar_template_matching" && has_images )
    {
      // Determine effective depth range for epipolar search
      double eff_min_depth = m_epipolar_min_depth;
      double eff_max_depth = m_epipolar_max_depth;

      if( m_epipolar_min_disparity > 0.0 && m_epipolar_max_disparity > 0.0 )
      {
        double fx = left_cam.get_intrinsics()->focal_length();
        double baseline = ( left_cam.center() - right_cam.center() ).norm();

        eff_min_depth = fx * baseline / m_epipolar_max_disparity;
        eff_max_depth = fx * baseline / m_epipolar_min_disparity;
      }

      // Compute epipolar points from camera geometry
      auto epipolar_head = compute_epipolar_points(
        left_cam, right_cam, result.left_head,
        eff_min_depth, eff_max_depth, m_epipolar_num_samples );
      auto epipolar_tail = compute_epipolar_points(
        left_cam, right_cam, result.left_tail,
        eff_min_depth, eff_max_depth, m_epipolar_num_samples );

      // Branch on descriptor type for the actual matching
      bool descriptor_available = false;
      std::string descriptor_label;

      if( m_epipolar_descriptor_type == "ncc" )
      {
        descriptor_available = true;
        descriptor_label = "ncc";

        // Convert unrectified images to grayscale cv::Mat
        cv::Mat left_gray = kwiver::arrows::ocv::image_container::vital_to_ocv(
          left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
        cv::Mat right_gray = kwiver::arrows::ocv::image_container::vital_to_ocv(
          right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

        if( left_gray.channels() == 3 )
          cv::cvtColor( left_gray, left_gray, cv::COLOR_BGR2GRAY );
        else if( left_gray.channels() == 4 )
          cv::cvtColor( left_gray, left_gray, cv::COLOR_BGRA2GRAY );

        if( right_gray.channels() == 3 )
          cv::cvtColor( right_gray, right_gray, cv::COLOR_BGR2GRAY );
        else if( right_gray.channels() == 4 )
          cv::cvtColor( right_gray, right_gray, cv::COLOR_BGRA2GRAY );

        auto t_ncc_start = std::chrono::steady_clock::now();

        head_found = find_corresponding_point_epipolar_template_matching(
          left_gray, right_gray, result.left_head, epipolar_head, result.right_head );
        tail_found = find_corresponding_point_epipolar_template_matching(
          left_gray, right_gray, result.left_tail, epipolar_tail, result.right_tail );

        auto t_ncc_end = std::chrono::steady_clock::now();
        LOG_INFO( logger, "NCC point-by-point ("
          << epipolar_head.size() << "+" << epipolar_tail.size() << " pts): "
          << std::chrono::duration_cast< std::chrono::milliseconds >(
               t_ncc_end - t_ncc_start ).count() << "ms" );
      }
      else if( m_epipolar_descriptor_type == "ncc_strip" )
      {
        descriptor_available = true;
        descriptor_label = "ncc_strip";

        // Convert unrectified images to grayscale cv::Mat
        cv::Mat left_gray = kwiver::arrows::ocv::image_container::vital_to_ocv(
          left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
        cv::Mat right_gray = kwiver::arrows::ocv::image_container::vital_to_ocv(
          right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

        if( left_gray.channels() == 3 )
          cv::cvtColor( left_gray, left_gray, cv::COLOR_BGR2GRAY );
        else if( left_gray.channels() == 4 )
          cv::cvtColor( left_gray, left_gray, cv::COLOR_BGRA2GRAY );

        if( right_gray.channels() == 3 )
          cv::cvtColor( right_gray, right_gray, cv::COLOR_BGR2GRAY );
        else if( right_gray.channels() == 4 )
          cv::cvtColor( right_gray, right_gray, cv::COLOR_BGRA2GRAY );

        auto t_strip_start = std::chrono::steady_clock::now();

        head_found = find_corresponding_point_epipolar_strip_ncc(
          left_gray, right_gray, result.left_head, epipolar_head, result.right_head );
        tail_found = find_corresponding_point_epipolar_strip_ncc(
          left_gray, right_gray, result.left_tail, epipolar_tail, result.right_tail );

        auto t_strip_end = std::chrono::steady_clock::now();
        LOG_INFO( logger, "Strip NCC (2 kps): "
          << std::chrono::duration_cast< std::chrono::milliseconds >(
               t_strip_end - t_strip_start ).count() << "ms" );
      }
#ifdef VIAME_ENABLE_PYTHON
      else if( m_epipolar_descriptor_type == "dino" )
      {
        descriptor_label = "dino";

        if( !dino_ensure_initialized(
              m_dino_model_name, m_dino_threshold, m_dino_weights_path ) )
        {
          throw std::runtime_error(
            "DINO matcher failed to initialize. "
            "Ensure viame.pytorch.dino_matcher is installed and PyTorch is available." );
        }
        else
        {
          // Convert images to BGR cv::Mat for DINO feature extraction
          cv::Mat left_bgr = kwiver::arrows::ocv::image_container::vital_to_ocv(
            left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
          cv::Mat right_bgr = kwiver::arrows::ocv::image_container::vital_to_ocv(
            right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

          // Use cropped images for DINO if crop is active
          cv::Mat dino_left = m_dino_crop_active ? m_dino_left_cropped : left_bgr;
          cv::Mat dino_right = m_dino_crop_active ? m_dino_right_cropped : right_bgr;

          // Only call set_images when the frame changes
          int64_t cur_frame = static_cast< int64_t >( m_cached_frame_id );
          if( cur_frame != s_dino_cached_frame_id )
          {
            auto t_dino_ext_start = std::chrono::steady_clock::now();

            if( dino_set_images( dino_left, dino_right ) )
            {
              s_dino_cached_frame_id = cur_frame;
            }
            else
            {
              s_dino_cached_frame_id = -1;
            }

            auto t_dino_ext_end = std::chrono::steady_clock::now();
            LOG_INFO( logger, "DINO extraction: "
              << std::chrono::duration_cast< std::chrono::milliseconds >(
                   t_dino_ext_end - t_dino_ext_start ).count() << "ms ("
              << dino_left.cols << "x" << dino_left.rows
              << ( m_dino_crop_active ? " crop" : " full" ) << ")" );
          }

          if( s_dino_cached_frame_id >= 0 )
          {
            descriptor_available = true;

            // Compute crop offsets for DINO coordinate transformation
            double left_off_x = m_dino_crop_active ? m_dino_left_crop.x : 0;
            double left_off_y = m_dino_crop_active ? m_dino_left_crop.y : 0;
            double right_off_x = m_dino_crop_active ? m_dino_right_crop.x : 0;
            double right_off_y = m_dino_crop_active ? m_dino_right_crop.y : 0;

            if( m_dino_top_k > 0 )
            {
              // Two-stage: DINO top-K filtering + NCC refinement
              // Prepare grayscale images for NCC (always full resolution)
              cv::Mat left_gray, right_gray;
              if( left_bgr.channels() == 3 )
                cv::cvtColor( left_bgr, left_gray, cv::COLOR_BGR2GRAY );
              else if( left_bgr.channels() == 4 )
                cv::cvtColor( left_bgr, left_gray, cv::COLOR_BGRA2GRAY );
              else
                left_gray = left_bgr;

              if( right_bgr.channels() == 3 )
                cv::cvtColor( right_bgr, right_gray, cv::COLOR_BGR2GRAY );
              else if( right_bgr.channels() == 4 )
                cv::cvtColor( right_bgr, right_gray, cv::COLOR_BGRA2GRAY );
              else
                right_gray = right_bgr;

              // Head: offset coords for DINO, get top-K, NCC uses original coords
              std::vector< kv::vector_2d > dino_epipolar_head;
              dino_epipolar_head.reserve( epipolar_head.size() );
              for( const auto& pt : epipolar_head )
              {
                dino_epipolar_head.push_back(
                  kv::vector_2d( pt.x() - right_off_x, pt.y() - right_off_y ) );
              }

              auto t_dino_match_start = std::chrono::steady_clock::now();

              auto head_indices = dino_get_top_k_indices(
                result.left_head.x() - left_off_x, result.left_head.y() - left_off_y,
                dino_epipolar_head, m_dino_top_k );

              // Tail: same approach
              std::vector< kv::vector_2d > dino_epipolar_tail;
              dino_epipolar_tail.reserve( epipolar_tail.size() );
              for( const auto& pt : epipolar_tail )
              {
                dino_epipolar_tail.push_back(
                  kv::vector_2d( pt.x() - right_off_x, pt.y() - right_off_y ) );
              }

              auto tail_indices = dino_get_top_k_indices(
                result.left_tail.x() - left_off_x, result.left_tail.y() - left_off_y,
                dino_epipolar_tail, m_dino_top_k );

              auto t_dino_match_end = std::chrono::steady_clock::now();
              LOG_INFO( logger, "DINO matching (2 kps): "
                << std::chrono::duration_cast< std::chrono::milliseconds >(
                     t_dino_match_end - t_dino_match_start ).count() << "ms" );

              auto t_ncc_refine_start = std::chrono::steady_clock::now();

              if( !head_indices.empty() )
              {
                // Indices map 1:1 to original epipolar arrays
                std::vector< kv::vector_2d > filtered_head;
                filtered_head.reserve( head_indices.size() );
                for( int idx : head_indices )
                {
                  filtered_head.push_back( epipolar_head[idx] );
                }
                head_found = find_corresponding_point_epipolar_template_matching(
                  left_gray, right_gray, result.left_head,
                  filtered_head, result.right_head );
              }

              if( !tail_indices.empty() )
              {
                std::vector< kv::vector_2d > filtered_tail;
                filtered_tail.reserve( tail_indices.size() );
                for( int idx : tail_indices )
                {
                  filtered_tail.push_back( epipolar_tail[idx] );
                }
                tail_found = find_corresponding_point_epipolar_template_matching(
                  left_gray, right_gray, result.left_tail,
                  filtered_tail, result.right_tail );
              }

              auto t_ncc_refine_end = std::chrono::steady_clock::now();
              LOG_INFO( logger, "NCC refinement (top-K): "
                << std::chrono::duration_cast< std::chrono::milliseconds >(
                     t_ncc_refine_end - t_ncc_refine_start ).count() << "ms" );
            }
            else
            {
              // DINO-only mode (no NCC refinement)
              // Offset epipolar points for DINO coordinate space
              std::vector< kv::vector_2d > dino_epi_head, dino_epi_tail;
              dino_epi_head.reserve( epipolar_head.size() );
              dino_epi_tail.reserve( epipolar_tail.size() );
              for( const auto& pt : epipolar_head )
                dino_epi_head.push_back( kv::vector_2d( pt.x() - right_off_x, pt.y() - right_off_y ) );
              for( const auto& pt : epipolar_tail )
                dino_epi_tail.push_back( kv::vector_2d( pt.x() - right_off_x, pt.y() - right_off_y ) );

              auto head_match = dino_match_point(
                result.left_head.x() - left_off_x, result.left_head.y() - left_off_y,
                dino_epi_head, m_dino_threshold );
              auto tail_match = dino_match_point(
                result.left_tail.x() - left_off_x, result.left_tail.y() - left_off_y,
                dino_epi_tail, m_dino_threshold );

              head_found = head_match.success;
              tail_found = tail_match.success;

              // Convert back to full-image coordinates
              if( head_found )
              {
                result.right_head = kv::vector_2d(
                  head_match.x + right_off_x, head_match.y + right_off_y );
              }
              if( tail_found )
              {
                result.right_tail = kv::vector_2d(
                  tail_match.x + right_off_x, tail_match.y + right_off_y );
              }
            }
          }
        }
      }
#endif // VIAME_ENABLE_PYTHON

      // Debug: write images with epipolar curves overlaid
      if( descriptor_available && !m_debug_epipolar_directory.empty() )
      {
        cv::Mat left_color = kwiver::arrows::ocv::image_container::vital_to_ocv(
          left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
        cv::Mat right_color = kwiver::arrows::ocv::image_container::vital_to_ocv(
          right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

        if( left_color.channels() == 1 )
          cv::cvtColor( left_color, left_color, cv::COLOR_GRAY2BGR );
        else if( left_color.channels() == 4 )
          cv::cvtColor( left_color, left_color, cv::COLOR_BGRA2BGR );

        if( right_color.channels() == 1 )
          cv::cvtColor( right_color, right_color, cv::COLOR_GRAY2BGR );
        else if( right_color.channels() == 4 )
          cv::cvtColor( right_color, right_color, cv::COLOR_BGRA2BGR );

        struct debug_kp
        {
          const char* label;
          const kv::vector_2d& src_pt;
          const std::vector< kv::vector_2d >& epi_pts;
          bool found;
          const kv::vector_2d& match_pt;
        };

        debug_kp keypoints[2] = {
          { "head", result.left_head, epipolar_head, head_found, result.right_head },
          { "tail", result.left_tail, epipolar_tail, tail_found, result.right_tail }
        };

        for( int ki = 0; ki < 2; ++ki )
        {
          const auto& kp = keypoints[ki];

          cv::Mat left_draw = left_color.clone();
          cv::Mat right_draw = right_color.clone();

          cv::Point src_px( static_cast<int>( kp.src_pt.x() + 0.5 ),
                            static_cast<int>( kp.src_pt.y() + 0.5 ) );
          cv::circle( left_draw, src_px, 8, cv::Scalar( 255, 255, 0 ), 2 );
          cv::line( left_draw, src_px - cv::Point( 12, 0 ),
                    src_px + cv::Point( 12, 0 ), cv::Scalar( 255, 255, 0 ), 1 );
          cv::line( left_draw, src_px - cv::Point( 0, 12 ),
                    src_px + cv::Point( 0, 12 ), cv::Scalar( 255, 255, 0 ), 1 );

          if( kp.epi_pts.size() >= 2 )
          {
            std::vector< cv::Point > poly;
            poly.reserve( kp.epi_pts.size() );
            for( const auto& ep : kp.epi_pts )
            {
              poly.emplace_back( static_cast<int>( ep.x() + 0.5 ),
                                 static_cast<int>( ep.y() + 0.5 ) );
            }
            cv::polylines( right_draw, poly, false, cv::Scalar( 0, 255, 0 ), 2 );
            cv::circle( right_draw, poly.front(), 6, cv::Scalar( 0, 200, 255 ), 2 );
            cv::circle( right_draw, poly.back(), 6, cv::Scalar( 255, 0, 200 ), 2 );
          }

          if( kp.found )
          {
            cv::Point match_px( static_cast<int>( kp.match_pt.x() + 0.5 ),
                                static_cast<int>( kp.match_pt.y() + 0.5 ) );
            cv::circle( right_draw, match_px, 8, cv::Scalar( 0, 0, 255 ), 2 );
            cv::line( right_draw, match_px - cv::Point( 12, 0 ),
                      match_px + cv::Point( 12, 0 ), cv::Scalar( 0, 0, 255 ), 1 );
            cv::line( right_draw, match_px - cv::Point( 0, 12 ),
                      match_px + cv::Point( 0, 12 ), cv::Scalar( 0, 0, 255 ), 1 );
          }

          cv::Mat canvas;
          cv::hconcat( left_draw, right_draw, canvas );

          std::string status = kp.found ? "MATCHED" : "NO MATCH";
          std::string label = descriptor_label + " " + kp.label + " - " + status +
            " (" + std::to_string( kp.epi_pts.size() ) + " samples)";
          cv::putText( canvas, label, cv::Point( 10, 30 ),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar( 0, 255, 255 ), 2 );

          std::string filename = m_debug_epipolar_directory + "/epipolar_" +
            std::to_string( m_debug_frame_counter ) + "_" + kp.label + ".jpg";
          cv::imwrite( filename, canvas );
        }

        m_debug_frame_counter++;
      }

      if( descriptor_available && ( head_found || tail_found ) )
      {
        result.method_used = "epipolar_template_matching";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
#endif
    else if( method == "feature_descriptor" && left_image && right_image )
    {
      kv::vector_2d left_head_copy = result.left_head;
      kv::vector_2d left_tail_copy = result.left_tail;

      head_found = find_corresponding_point_feature_descriptor(
        left_image, right_image, left_head_copy, result.right_head,
        &left_cam, &right_cam );
      tail_found = find_corresponding_point_feature_descriptor(
        left_image, right_image, left_tail_copy, result.right_tail,
        &left_cam, &right_cam );

      if( head_found || tail_found )
      {
        if( head_found )
          result.left_head = left_head_copy;
        if( tail_found )
          result.left_tail = left_tail_copy;
        result.method_used = "feature_descriptor";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
    else if( method == "ransac_feature" && left_image && right_image )
    {
      kv::vector_2d left_head_copy = result.left_head;
      kv::vector_2d left_tail_copy = result.left_tail;

      head_found = find_corresponding_point_ransac_feature(
        left_image, right_image, left_head_copy, result.right_head,
        &left_cam, &right_cam );
      tail_found = find_corresponding_point_ransac_feature(
        left_image, right_image, left_tail_copy, result.right_tail,
        &left_cam, &right_cam );

      if( head_found || tail_found )
      {
        if( head_found )
          result.left_head = left_head_copy;
        if( tail_found )
          result.left_tail = left_tail_copy;
        result.method_used = "ransac_feature";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
  }

  result.head_found = head_found;
  result.tail_found = tail_found;
  result.success = ( head_found || tail_found );
  return result;
}

#ifdef VIAME_ENABLE_OPENCV
// -----------------------------------------------------------------------------
map_keypoints_to_camera::stereo_image_data
map_keypoints_to_camera
::prepare_stereo_images(
  const std::vector< std::string >& methods,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image )
{
  stereo_image_data data;
  data.rectified_available = false;
  data.disparity_available = false;

  if( !left_image || !right_image )
  {
    return data;
  }

  // Check which methods need rectified images
  bool needs_rectified = false;
  for( const auto& method : methods )
  {
    if( method == "template_matching" || method == "compute_disparity" )
    {
      needs_rectified = true;
    }
  }

  if( !needs_rectified )
  {
    return data;
  }

  // Convert to OpenCV format
  cv::Mat left_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat right_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale (handle each image independently since they may have different channel counts)
  if( left_cv.channels() == 3 )
  {
    cv::cvtColor( left_cv, left_cv, cv::COLOR_BGR2GRAY );
  }
  else if( left_cv.channels() == 4 )
  {
    cv::cvtColor( left_cv, left_cv, cv::COLOR_BGRA2GRAY );
  }
  // If already 1 channel, no conversion needed

  if( right_cv.channels() == 3 )
  {
    cv::cvtColor( right_cv, right_cv, cv::COLOR_BGR2GRAY );
  }
  else if( right_cv.channels() == 4 )
  {
    cv::cvtColor( right_cv, right_cv, cv::COLOR_BGRA2GRAY );
  }
  // If already 1 channel, no conversion needed

  // Compute rectification maps if needed
  compute_rectification_maps( left_cam, right_cam, left_cv.size() );

  // Rectify images
  data.left_rectified = rectify_image( left_cv, false );
  data.right_rectified = rectify_image( right_cv, true );
  data.rectified_available = true;

  // Compute disparity if needed for template matching disparity hint
  if( m_use_disparity_hint && m_stereo_depth_map_algorithm )
  {
    data.disparity_map = compute_sgbm_disparity( data.left_rectified, data.right_rectified );
    data.disparity_available = !data.disparity_map.empty();
  }

  return data;
}
#endif

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_feature_descriptor(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  kv::vector_2d& left_point,
  kv::vector_2d& right_point,
  const kv::simple_camera_perspective* left_cam,
  const kv::simple_camera_perspective* right_cam )
{
  if( !m_feature_detector || !m_descriptor_extractor || !m_feature_matcher )
  {
    return false;
  }

  // Detect features and extract descriptors if not cached for this frame
  if( !m_cached_left_features || !m_cached_right_features )
  {
    m_cached_left_features = m_feature_detector->detect( left_image );
    m_cached_right_features = m_feature_detector->detect( right_image );

    m_cached_left_descriptors = m_descriptor_extractor->extract(
      left_image, m_cached_left_features );
    m_cached_right_descriptors = m_descriptor_extractor->extract(
      right_image, m_cached_right_features );

    m_cached_matches = m_feature_matcher->match(
      m_cached_left_features, m_cached_left_descriptors,
      m_cached_right_features, m_cached_right_descriptors );
  }

  if( !m_cached_matches || m_cached_matches->size() == 0 )
  {
    return false;
  }

  // Get the feature vectors
  auto left_features = m_cached_left_features->features();
  auto right_features = m_cached_right_features->features();
  auto matches = m_cached_matches->matches();

  // Compute expected right point location using depth projection if enabled
  kv::vector_2d expected_right_point = left_point;  // Default: same as left point
  if( m_use_disparity_aware_feature_search && left_cam && right_cam )
  {
    // Use feature_search_depth if valid, otherwise fall back to default_depth
    double search_depth = ( m_feature_search_depth > 0 ) ? m_feature_search_depth : m_default_depth;
    expected_right_point = viame::core::project_left_to_right( *left_cam, *right_cam, left_point, search_depth );
  }

  // Find the closest matched feature to our query point
  // For left features: search near left_point
  // For right features: search near expected_right_point (disparity-aware)
  double best_dist = std::numeric_limits<double>::max();
  kv::vector_2d best_left_point;
  kv::vector_2d best_right_point;
  bool found = false;

  for( const auto& match : matches )
  {
    if( match.first >= left_features.size() ||
        match.second >= right_features.size() )
    {
      continue;
    }

    const auto& left_feat = left_features[match.first];
    const auto& right_feat = right_features[match.second];

    kv::vector_2d left_feat_loc = left_feat->loc();
    kv::vector_2d right_feat_loc = right_feat->loc();

    // Check if left feature is within search radius of query point
    double left_dist = ( left_feat_loc - left_point ).norm();
    if( left_dist >= m_feature_search_radius )
    {
      continue;
    }

    // Check if right feature is within search radius of expected location
    double right_dist = ( right_feat_loc - expected_right_point ).norm();
    if( right_dist >= m_feature_search_radius )
    {
      continue;
    }

    // Use combined distance metric (sum of left and right distances)
    double combined_dist = left_dist + right_dist;
    if( combined_dist < best_dist )
    {
      best_dist = combined_dist;
      best_left_point = left_feat_loc;
      best_right_point = right_feat_loc;
      found = true;
    }
  }

  if( found )
  {
    // Apply local offset: displacement from left feature → left keypoint
    // approximates displacement from right feature → right keypoint
    right_point = best_right_point + ( left_point - best_left_point );
    // Don't modify left_point — preserve the original annotated keypoint
  }

  return found;
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_ransac_feature(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  kv::vector_2d& left_point,
  kv::vector_2d& right_point,
  const kv::simple_camera_perspective* left_cam,
  const kv::simple_camera_perspective* right_cam )
{
  if( !m_feature_detector || !m_descriptor_extractor ||
      !m_feature_matcher || !m_fundamental_matrix_estimator )
  {
    return false;
  }

  // Detect features and extract descriptors if not cached for this frame
  if( !m_cached_left_features || !m_cached_right_features )
  {
    m_cached_left_features = m_feature_detector->detect( left_image );
    m_cached_right_features = m_feature_detector->detect( right_image );

    m_cached_left_descriptors = m_descriptor_extractor->extract(
      left_image, m_cached_left_features );
    m_cached_right_descriptors = m_descriptor_extractor->extract(
      right_image, m_cached_right_features );

    m_cached_matches = m_feature_matcher->match(
      m_cached_left_features, m_cached_left_descriptors,
      m_cached_right_features, m_cached_right_descriptors );
  }

  if( !m_cached_matches || m_cached_matches->size() == 0 )
  {
    return false;
  }

  // Get the feature vectors
  auto left_features = m_cached_left_features->features();
  auto right_features = m_cached_right_features->features();
  auto matches = m_cached_matches->matches();

  // Estimate fundamental matrix using RANSAC to filter outliers
  std::vector< bool > inliers;
  auto F = m_fundamental_matrix_estimator->estimate(
    m_cached_left_features, m_cached_right_features,
    m_cached_matches, inliers, m_ransac_inlier_scale );

  // Count inliers
  int inlier_count = 0;
  for( bool is_inlier : inliers )
  {
    if( is_inlier )
    {
      ++inlier_count;
    }
  }

  if( inlier_count < m_min_ransac_inliers )
  {
    return false;
  }

  // Compute expected right point location using depth projection if enabled
  kv::vector_2d expected_right_point = left_point;  // Default: same as left point
  if( m_use_disparity_aware_feature_search && left_cam && right_cam )
  {
    // Use feature_search_depth if valid, otherwise fall back to default_depth
    double search_depth = ( m_feature_search_depth > 0 ) ? m_feature_search_depth : m_default_depth;
    expected_right_point = viame::core::project_left_to_right( *left_cam, *right_cam, left_point, search_depth );
  }

  // Find the closest inlier match to our query point
  // For left features: search near left_point
  // For right features: search near expected_right_point (disparity-aware)
  double best_dist = std::numeric_limits<double>::max();
  kv::vector_2d best_left_point;
  kv::vector_2d best_right_point;
  bool found = false;

  for( size_t i = 0; i < matches.size(); ++i )
  {
    if( !inliers[i] )
    {
      continue;
    }

    const auto& match = matches[i];
    if( match.first >= left_features.size() ||
        match.second >= right_features.size() )
    {
      continue;
    }

    const auto& left_feat = left_features[match.first];
    const auto& right_feat = right_features[match.second];

    kv::vector_2d left_feat_loc = left_feat->loc();
    kv::vector_2d right_feat_loc = right_feat->loc();

    // Check if left feature is within search radius of query point
    double left_dist = ( left_feat_loc - left_point ).norm();
    if( left_dist >= m_feature_search_radius )
    {
      continue;
    }

    // Check if right feature is within search radius of expected location
    double right_dist = ( right_feat_loc - expected_right_point ).norm();
    if( right_dist >= m_feature_search_radius )
    {
      continue;
    }

    // Use combined distance metric (sum of left and right distances)
    double combined_dist = left_dist + right_dist;
    if( combined_dist < best_dist )
    {
      best_dist = combined_dist;
      best_left_point = left_feat_loc;
      best_right_point = right_feat_loc;
      found = true;
    }
  }

  if( found )
  {
    // Apply local offset: displacement from left feature → left keypoint
    // approximates displacement from right feature → right keypoint
    right_point = best_right_point + ( left_point - best_left_point );
    // Don't modify left_point — preserve the original annotated keypoint
  }

  return found;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::clear_feature_cache()
{
  m_cached_left_features.reset();
  m_cached_right_features.reset();
  m_cached_left_descriptors.reset();
  m_cached_right_descriptors.reset();
  m_cached_matches.reset();
  m_cached_compute_disparity.reset();
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_frame_id( kv::frame_id_t frame_id )
{
  if( m_cached_frame_id != frame_id )
  {
    clear_feature_cache();
    m_cached_frame_id = frame_id;
  }
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
map_keypoints_to_camera
::get_cached_disparity() const
{
  return m_cached_compute_disparity;
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
map_keypoints_to_camera
::get_cached_rectified_left() const
{
#ifdef VIAME_ENABLE_OPENCV
  if( m_cached_stereo_images.rectified_available &&
      !m_cached_stereo_images.left_rectified.empty() )
  {
    return std::make_shared< kwiver::arrows::ocv::image_container >(
      m_cached_stereo_images.left_rectified,
      kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );
  }
#endif
  return nullptr;
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
map_keypoints_to_camera
::get_cached_rectified_right() const
{
#ifdef VIAME_ENABLE_OPENCV
  if( m_cached_stereo_images.rectified_available &&
      !m_cached_stereo_images.right_rectified.empty() )
  {
    return std::make_shared< kwiver::arrows::ocv::image_container >(
      m_cached_stereo_images.right_rectified,
      kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );
  }
#endif
  return nullptr;
}

#ifdef VIAME_ENABLE_OPENCV

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::compute_rectification_maps(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const cv::Size& image_size )
{
  if( m_rectification_computed )
  {
    return;
  }

  // Get camera intrinsics
  auto left_intrinsics = left_cam.get_intrinsics();
  auto right_intrinsics = right_cam.get_intrinsics();

  // Convert to OpenCV matrices
  cv::Mat K1, K2, D1, D2, R, T;

  // Camera matrices
  Eigen::Matrix3d K1_eigen = left_intrinsics->as_matrix();
  Eigen::Matrix3d K2_eigen = right_intrinsics->as_matrix();
  cv::eigen2cv( K1_eigen, K1 );
  cv::eigen2cv( K2_eigen, K2 );

  // Distortion coefficients
  D1 = cv::Mat::zeros( 5, 1, CV_64F );
  D2 = cv::Mat::zeros( 5, 1, CV_64F );

  if( m_use_distortion )
  {
    std::vector< double > left_dist = left_intrinsics->dist_coeffs();
    std::vector< double > right_dist = right_intrinsics->dist_coeffs();

    // Convert distortion coefficients to OpenCV format
    for( size_t i = 0; i < std::min( left_dist.size(), size_t(5) ); ++i )
    {
      D1.at< double >( static_cast< int >( i ), 0 ) = left_dist[i];
    }

    for( size_t i = 0; i < std::min( right_dist.size(), size_t(5) ); ++i )
    {
      D2.at< double >( static_cast< int >( i ), 0 ) = right_dist[i];
    }
  }

  // Compute rotation and translation from left camera frame to right camera frame
  // X_right = R_relative * X_left + t_relative
  Eigen::Matrix3d R_left = left_cam.rotation().matrix();
  Eigen::Matrix3d R_right = right_cam.rotation().matrix();
  Eigen::Matrix3d R_relative = R_right * R_left.transpose();

  // Translation: t = R_right * (C_left - C_right)
  Eigen::Vector3d t_relative = R_right * ( left_cam.center() - right_cam.center() );

  cv::eigen2cv( R_relative, R );
  cv::eigen2cv( t_relative, T );

  // Compute rectification transforms
  cv::Mat Q;
  cv::stereoRectify( K1, D1, K2, D2, image_size, R, T,
                     m_R1, m_R2, m_P1, m_P2, Q,
                     cv::CALIB_ZERO_DISPARITY, 0 );

  // Store camera matrices and distortion coefficients
  m_K1 = K1.clone();
  m_K2 = K2.clone();
  m_D1 = D1.clone();
  m_D2 = D2.clone();

  // Compute rectification maps
  cv::initUndistortRectifyMap( K1, D1, m_R1, m_P1, image_size, CV_32FC1,
    m_rectification_map_left_x, m_rectification_map_left_y );
  cv::initUndistortRectifyMap( K2, D2, m_R2, m_P2, image_size, CV_32FC1,
    m_rectification_map_right_x, m_rectification_map_right_y );

  m_rectification_computed = true;
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::rectification_computed() const
{
  return m_rectification_computed;
}

// -----------------------------------------------------------------------------
kv::vector_2d
map_keypoints_to_camera
::rectify_point(
  const kv::vector_2d& original_point,
  bool is_right_camera ) const
{
  if( !m_rectification_computed )
  {
    return original_point;
  }

  const cv::Mat& K = is_right_camera ? m_K2 : m_K1;
  const cv::Mat& D = is_right_camera ? m_D2 : m_D1;
  const cv::Mat& R = is_right_camera ? m_R2 : m_R1;
  const cv::Mat& P = is_right_camera ? m_P2 : m_P1;

  if( K.empty() || R.empty() || P.empty() )
  {
    return original_point;
  }

  std::vector< cv::Point2f > pts_in = { cv::Point2f( original_point.x(), original_point.y() ) };
  std::vector< cv::Point2f > pts_out;

  cv::undistortPoints( pts_in, pts_out, K, D, R, P );

  return pts_out.empty() ? original_point : kv::vector_2d( pts_out[0].x, pts_out[0].y );
}

// -----------------------------------------------------------------------------
kv::vector_2d
map_keypoints_to_camera
::unrectify_point(
  const kv::vector_2d& rectified_point,
  bool is_right_camera,
  const kv::simple_camera_perspective& ) const
{
  if( !m_rectification_computed )
  {
    return rectified_point;
  }

  const cv::Mat& R = is_right_camera ? m_R2 : m_R1;
  const cv::Mat& P = is_right_camera ? m_P2 : m_P1;
  const cv::Mat& K = is_right_camera ? m_K2 : m_K1;
  const cv::Mat& D = is_right_camera ? m_D2 : m_D1;

  // Extract rectified camera intrinsics from P (3x4 projection matrix)
  double fx_rect = P.at< double >( 0, 0 );
  double fy_rect = P.at< double >( 1, 1 );
  double cx_rect = P.at< double >( 0, 2 );
  double cy_rect = P.at< double >( 1, 2 );

  // Convert rectified pixel to normalized rectified coordinates
  double x_norm_rect = ( rectified_point.x() - cx_rect ) / fx_rect;
  double y_norm_rect = ( rectified_point.y() - cy_rect ) / fy_rect;

  // Apply inverse rectification rotation to get normalized original coordinates
  cv::Mat pt_rect = ( cv::Mat_<double>( 3, 1 ) << x_norm_rect, y_norm_rect, 1.0 );
  cv::Mat pt_orig = R.t() * pt_rect;

  double x_norm = pt_orig.at< double >( 0, 0 ) / pt_orig.at< double >( 2, 0 );
  double y_norm = pt_orig.at< double >( 1, 0 ) / pt_orig.at< double >( 2, 0 );

  // Apply distortion and camera matrix using projectPoints with identity pose
  std::vector< cv::Point3f > pts_3d = { cv::Point3f( x_norm, y_norm, 1.0f ) };
  std::vector< cv::Point2f > pts_2d;
  cv::Mat rvec = cv::Mat::zeros( 3, 1, CV_64F );
  cv::Mat tvec = cv::Mat::zeros( 3, 1, CV_64F );

  cv::projectPoints( pts_3d, rvec, tvec, K, D, pts_2d );

  return pts_2d.empty() ? rectified_point : kv::vector_2d( pts_2d[0].x, pts_2d[0].y );
}

// -----------------------------------------------------------------------------
cv::Mat
map_keypoints_to_camera
::rectify_image( const cv::Mat& image, bool is_right_camera ) const
{
  if( !m_rectification_computed )
  {
    return image.clone();
  }

  cv::Mat rectified;
  if( is_right_camera )
  {
    cv::remap( image, rectified, m_rectification_map_right_x,
               m_rectification_map_right_y, cv::INTER_LINEAR );
  }
  else
  {
    cv::remap( image, rectified, m_rectification_map_left_x,
               m_rectification_map_left_y, cv::INTER_LINEAR );
  }
  return rectified;
}

// -----------------------------------------------------------------------------
// Helper function to compute census transform of an image
// Census transform compares each pixel to its neighbors, creating a binary pattern
// that is robust to illumination changes
namespace {
cv::Mat compute_census_transform( const cv::Mat& input, int window_radius = 2 )
{
  cv::Mat gray;
  if( input.channels() == 3 )
  {
    cv::cvtColor( input, gray, cv::COLOR_BGR2GRAY );
  }
  else
  {
    gray = input;
  }

  cv::Mat census( gray.size(), CV_32S, cv::Scalar( 0 ) );

  for( int y = window_radius; y < gray.rows - window_radius; ++y )
  {
    for( int x = window_radius; x < gray.cols - window_radius; ++x )
    {
      unsigned int census_val = 0;
      uchar center = gray.at< uchar >( y, x );
      int bit_pos = 0;

      for( int dy = -window_radius; dy <= window_radius; ++dy )
      {
        for( int dx = -window_radius; dx <= window_radius; ++dx )
        {
          if( dx == 0 && dy == 0 ) continue;  // Skip center

          if( gray.at< uchar >( y + dy, x + dx ) < center )
          {
            census_val |= ( 1u << bit_pos );
          }
          ++bit_pos;
        }
      }
      census.at< int >( y, x ) = static_cast< int >( census_val );
    }
  }

  return census;
}

// Compute Hamming distance between two census values
int census_hamming_distance( int a, int b )
{
  unsigned int xor_val = static_cast< unsigned int >( a ^ b );
  int dist = 0;
  while( xor_val )
  {
    dist += xor_val & 1;
    xor_val >>= 1;
  }
  return dist;
}

// Template matching using census transform (sum of Hamming distances)
// Returns correlation-like score (higher is better, normalized to 0-1 range)
double census_template_match(
  const cv::Mat& census_template,
  const cv::Mat& census_search,
  int search_x, int search_y,
  int template_width, int template_height )
{
  int max_distance = template_width * template_height * 24;  // 24 bits max per pixel for 5x5 window
  int total_distance = 0;

  for( int ty = 0; ty < template_height; ++ty )
  {
    for( int tx = 0; tx < template_width; ++tx )
    {
      int t_val = census_template.at< int >( ty, tx );
      int s_val = census_search.at< int >( search_y + ty, search_x + tx );
      total_distance += census_hamming_distance( t_val, s_val );
    }
  }

  // Convert to correlation-like score (1.0 = perfect match, 0.0 = worst)
  return 1.0 - static_cast< double >( total_distance ) / max_distance;
}
} // anonymous namespace

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::prepare_source_template(
  const cv::Mat& source_image, int x, int y,
  prepared_template& tmpl ) const
{
  tmpl.valid = false;
  int half_template = m_template_size / 2;
  int margin = m_use_census_transform ? half_template + 2 : half_template;

  if( x < margin || x >= source_image.cols - margin ||
      y < margin || y >= source_image.rows - margin )
  {
    return false;
  }

  // Extract NCC template
  cv::Rect template_rect( x - half_template, y - half_template,
                          m_template_size, m_template_size );
  tmpl.ncc_template = source_image( template_rect ).clone();

  if( m_use_census_transform )
  {
    int census_margin = 2;
    cv::Rect template_rect_ext( x - half_template - census_margin,
                                 y - half_template - census_margin,
                                 m_template_size + 2 * census_margin,
                                 m_template_size + 2 * census_margin );
    cv::Mat template_region = source_image( template_rect_ext );
    cv::Mat census_full = compute_census_transform( template_region, census_margin );
    cv::Rect valid_rect( census_margin, census_margin, m_template_size, m_template_size );
    tmpl.census_template = census_full( valid_rect ).clone();
  }

  tmpl.valid = true;
  return true;
}

// -----------------------------------------------------------------------------
double
map_keypoints_to_camera
::score_template_at_point(
  const prepared_template& tmpl,
  const cv::Mat& target_image, int x, int y ) const
{
  int half_template = m_template_size / 2;
  int margin = m_use_census_transform ? half_template + 2 : half_template;

  if( x < margin || x >= target_image.cols - margin ||
      y < margin || y >= target_image.rows - margin )
  {
    return -1.0;
  }

  if( m_use_census_transform )
  {
    int census_margin = 2;
    cv::Rect target_rect_ext( x - half_template - census_margin,
                               y - half_template - census_margin,
                               m_template_size + 2 * census_margin,
                               m_template_size + 2 * census_margin );
    cv::Mat target_region = target_image( target_rect_ext );
    cv::Mat census_target = compute_census_transform( target_region, census_margin );

    return census_template_match( tmpl.census_template, census_target,
                                   census_margin, census_margin,
                                   m_template_size, m_template_size );
  }
  else
  {
    cv::Rect target_rect( x - half_template, y - half_template,
                          m_template_size, m_template_size );
    cv::Mat target_patch = target_image( target_rect );

    cv::Mat result;
    cv::matchTemplate( target_patch, tmpl.ncc_template, result, cv::TM_CCOEFF_NORMED );
    return static_cast< double >( result.at< float >( 0, 0 ) );
  }
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_template_matching(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect,
  const cv::Mat& disparity_map ) const
{
  int half_template = m_template_size / 2;
  int x_left = static_cast< int >( left_point_rect.x() );
  int y_left = static_cast< int >( left_point_rect.y() );

  // Prepare source template (handles bounds checking and extraction)
  prepared_template tmpl;
  if( !prepare_source_template( left_image_rect, x_left, y_left, tmpl ) )
  {
    return false;
  }

  int margin = m_use_census_transform ? half_template + 2 : half_template;

  // Determine expected disparity using priority:
  // 1. Explicitly configured disparity (if > 0)
  // 2. SGBM disparity hint from disparity map (if enabled and available)
  // 3. Computed from default_depth using camera parameters
  double expected_disparity = 0.0;

  if( m_template_matching_disparity > 0 )
  {
    // Use explicitly configured disparity
    expected_disparity = m_template_matching_disparity;
  }
  else if( m_use_disparity_hint && !disparity_map.empty() )
  {
    // Sample SGBM disparity map near the query point
    // Average over a small window for robustness
    int window_size = 5;
    int half_window = window_size / 2;
    double disparity_sum = 0.0;
    int valid_count = 0;

    for( int dy = -half_window; dy <= half_window; ++dy )
    {
      for( int dx = -half_window; dx <= half_window; ++dx )
      {
        int sample_x = x_left + dx;
        int sample_y = y_left + dy;

        if( sample_x >= 0 && sample_x < disparity_map.cols &&
            sample_y >= 0 && sample_y < disparity_map.rows )
        {
          short disp_raw = disparity_map.at< short >( sample_y, sample_x );
          // SGBM returns fixed-point values scaled by 16, invalid values are negative
          if( disp_raw > 0 )
          {
            disparity_sum += static_cast< double >( disp_raw ) / 16.0;
            ++valid_count;
          }
        }
      }
    }

    if( valid_count > 0 )
    {
      expected_disparity = disparity_sum / valid_count;
    }
    else if( !m_P2.empty() && m_default_depth > 0 )
    {
      // Fall back to default depth computation
      expected_disparity = -m_P2.at< double >( 0, 3 ) / m_default_depth;
    }
  }
  else if( !m_P2.empty() && m_default_depth > 0 )
  {
    // Compute disparity from default depth using camera parameters
    expected_disparity = -m_P2.at< double >( 0, 3 ) / m_default_depth;
  }

  // Compute expected right x position based on disparity
  int expected_right_x = static_cast< int >( x_left - expected_disparity );

  // Define search region centered around expected position
  // Use half the search range on each side of expected position for efficiency
  int half_search = m_search_range / 2;
  int search_min_x = std::max( margin, expected_right_x - half_search );
  int search_max_x = std::min( right_image_rect.cols - margin, expected_right_x + half_search );

  // Ensure we don't search past the left point (disparity is always positive in standard stereo)
  search_max_x = std::min( search_max_x, x_left );

  if( search_max_x <= search_min_x )
  {
    return false;
  }

  // Determine vertical search range based on epipolar band setting
  int search_min_y = y_left - m_epipolar_band_halfwidth;
  int search_max_y = y_left + m_epipolar_band_halfwidth;

  // Clamp to valid image bounds
  search_min_y = std::max( margin, search_min_y );
  search_max_y = std::min( right_image_rect.rows - margin, search_max_y );

  if( search_max_y < search_min_y )
  {
    return false;
  }

  double max_val = -1.0;
  cv::Point max_loc( 0, 0 );

  if( m_use_census_transform )
  {
    // Census transform based matching (uses prepared census template)
    int census_margin = 2;

    // Compute census transform of search region
    cv::Rect search_rect_ext( search_min_x - half_template - census_margin,
                               search_min_y - half_template - census_margin,
                               ( search_max_x - search_min_x ) + m_template_size + 2 * census_margin,
                               ( search_max_y - search_min_y ) + m_template_size + 2 * census_margin );

    // Bounds check
    if( search_rect_ext.x < 0 || search_rect_ext.y < 0 ||
        search_rect_ext.x + search_rect_ext.width > right_image_rect.cols ||
        search_rect_ext.y + search_rect_ext.height > right_image_rect.rows )
    {
      return false;
    }

    cv::Mat search_region = right_image_rect( search_rect_ext );
    cv::Mat census_search = compute_census_transform( search_region, census_margin );

    // Search over the valid region
    int result_width = search_max_x - search_min_x + 1;
    int result_height = search_max_y - search_min_y + 1;

    for( int sy = 0; sy < result_height; ++sy )
    {
      for( int sx = 0; sx < result_width; ++sx )
      {
        double score = census_template_match( tmpl.census_template, census_search,
                                               sx + census_margin, sy + census_margin,
                                               m_template_size, m_template_size );
        if( score > max_val )
        {
          max_val = score;
          max_loc.x = sx;
          max_loc.y = sy;
        }
      }
    }

    // Convert max_loc to image coordinates
    right_point_rect = kv::vector_2d(
      search_min_x + max_loc.x,
      search_min_y + max_loc.y );
  }
  else
  {
    // Standard intensity-based template matching (uses prepared NCC template)

    // Define search region including epipolar band
    int search_height = ( search_max_y - search_min_y ) + m_template_size;
    cv::Rect search_rect( search_min_x - half_template,
                          search_min_y - half_template,
                          search_max_x - search_min_x + m_template_size,
                          search_height );

    // Check search rect validity
    if( search_rect.x < 0 || search_rect.y < 0 ||
        search_rect.x + search_rect.width > right_image_rect.cols ||
        search_rect.y + search_rect.height > right_image_rect.rows )
    {
      return false;
    }

    cv::Mat search_region = right_image_rect( search_rect );
    cv::Mat result;

    if( m_use_multires_search && search_rect.width > m_template_size + m_multires_coarse_step * 4 )
    {
      // Multi-resolution search: coarse pass then fine pass
      cv::matchTemplate( search_region, tmpl.ncc_template, result, cv::TM_CCOEFF_NORMED );

      // Find best match in coarse grid
      double coarse_max_val = -1.0;
      cv::Point coarse_max_loc( 0, 0 );

      for( int ry = 0; ry < result.rows; ++ry )
      {
        for( int rx = 0; rx < result.cols; rx += m_multires_coarse_step )
        {
          double val = result.at< float >( ry, rx );
          if( val > coarse_max_val )
          {
            coarse_max_val = val;
            coarse_max_loc.x = rx;
            coarse_max_loc.y = ry;
          }
        }
      }

      // Fine pass: search around the coarse best match
      int fine_half_range = m_multires_coarse_step * 2;
      int fine_min_x = std::max( 0, coarse_max_loc.x - fine_half_range );
      int fine_max_x = std::min( result.cols - 1, coarse_max_loc.x + fine_half_range );
      int fine_min_y = std::max( 0, coarse_max_loc.y - fine_half_range );
      int fine_max_y = std::min( result.rows - 1, coarse_max_loc.y + fine_half_range );

      max_val = coarse_max_val;
      max_loc = coarse_max_loc;

      for( int ry = fine_min_y; ry <= fine_max_y; ++ry )
      {
        for( int rx = fine_min_x; rx <= fine_max_x; ++rx )
        {
          double val = result.at< float >( ry, rx );
          if( val > max_val )
          {
            max_val = val;
            max_loc.x = rx;
            max_loc.y = ry;
          }
        }
      }
    }
    else
    {
      // Standard single-pass template matching
      cv::matchTemplate( search_region, tmpl.ncc_template, result, cv::TM_CCOEFF_NORMED );

      // Find best match
      double min_val;
      cv::Point min_loc;
      cv::minMaxLoc( result, &min_val, &max_val, &min_loc, &max_loc );
    }

    // Convert max_loc to image coordinates
    right_point_rect = kv::vector_2d(
      search_rect.x + max_loc.x + half_template,
      search_rect.y + max_loc.y + half_template );
  }

  // Use a threshold for match quality
  if( max_val < m_template_matching_threshold )
  {
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_epipolar_template_matching(
  const cv::Mat& source_image,
  const cv::Mat& target_image,
  const kv::vector_2d& source_point,
  const std::vector< kv::vector_2d >& epipolar_points,
  kv::vector_2d& target_point ) const
{
  if( epipolar_points.empty() )
  {
    return false;
  }

  int x_src = static_cast< int >( source_point.x() + 0.5 );
  int y_src = static_cast< int >( source_point.y() + 0.5 );

  prepared_template tmpl;
  if( !prepare_source_template( source_image, x_src, y_src, tmpl ) )
  {
    return false;
  }

  double best_score = -1.0;
  kv::vector_2d best_point;

  for( const auto& ep_pt : epipolar_points )
  {
    int x_tgt = static_cast< int >( ep_pt.x() + 0.5 );
    int y_tgt = static_cast< int >( ep_pt.y() + 0.5 );

    double score = score_template_at_point( tmpl, target_image, x_tgt, y_tgt );
    if( score > best_score )
    {
      best_score = score;
      best_point = ep_pt;
    }
  }

  if( best_score < m_template_matching_threshold )
  {
    return false;
  }

  target_point = best_point;
  return true;
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_epipolar_strip_ncc(
  const cv::Mat& source_image,
  const cv::Mat& target_image,
  const kv::vector_2d& source_point,
  const std::vector< kv::vector_2d >& epipolar_points,
  kv::vector_2d& target_point ) const
{
  if( epipolar_points.empty() )
  {
    return false;
  }

  int x_src = static_cast< int >( source_point.x() + 0.5 );
  int y_src = static_cast< int >( source_point.y() + 0.5 );

  prepared_template tmpl;
  if( !prepare_source_template( source_image, x_src, y_src, tmpl ) )
  {
    return false;
  }

  int half_template = m_template_size / 2;

  // Compute bounding box of all epipolar points
  double min_x = epipolar_points[0].x();
  double max_x = min_x;
  double min_y = epipolar_points[0].y();
  double max_y = min_y;

  for( const auto& pt : epipolar_points )
  {
    min_x = std::min( min_x, pt.x() );
    max_x = std::max( max_x, pt.x() );
    min_y = std::min( min_y, pt.y() );
    max_y = std::max( max_y, pt.y() );
  }

  // Expand by half_template so the template can be centered at any epipolar point
  int strip_x = static_cast< int >( std::floor( min_x ) ) - half_template;
  int strip_y = static_cast< int >( std::floor( min_y ) ) - half_template;
  int strip_x2 = static_cast< int >( std::ceil( max_x ) ) + half_template;
  int strip_y2 = static_cast< int >( std::ceil( max_y ) ) + half_template;

  // Clamp to image bounds
  strip_x = std::max( 0, strip_x );
  strip_y = std::max( 0, strip_y );
  strip_x2 = std::min( target_image.cols - 1, strip_x2 );
  strip_y2 = std::min( target_image.rows - 1, strip_y2 );

  int strip_w = strip_x2 - strip_x + 1;
  int strip_h = strip_y2 - strip_y + 1;

  // Strip must be at least as large as the template
  if( strip_w < m_template_size || strip_h < m_template_size )
  {
    return false;
  }

  // Extract strip subimage and run matchTemplate
  cv::Rect strip_rect( strip_x, strip_y, strip_w, strip_h );
  cv::Mat strip = target_image( strip_rect );

  cv::Mat result;
  cv::matchTemplate( strip, tmpl.ncc_template, result, cv::TM_CCOEFF_NORMED );

  // Find maximum in result
  double max_val;
  cv::Point max_loc;
  cv::minMaxLoc( result, nullptr, &max_val, nullptr, &max_loc );

  if( max_val < m_template_matching_threshold )
  {
    return false;
  }

  // Convert result location to image coordinates
  // matchTemplate result offset is top-left of the template placement
  double match_x = strip_x + max_loc.x + half_template;
  double match_y = strip_y + max_loc.y + half_template;

  // Snap to the nearest epipolar point for geometric consistency
  double best_dist_sq = std::numeric_limits< double >::max();
  int best_idx = 0;

  for( int i = 0; i < static_cast< int >( epipolar_points.size() ); ++i )
  {
    double dx = epipolar_points[i].x() - match_x;
    double dy = epipolar_points[i].y() - match_y;
    double dist_sq = dx * dx + dy * dy;
    if( dist_sq < best_dist_sq )
    {
      best_dist_sq = dist_sq;
      best_idx = i;
    }
  }

  target_point = epipolar_points[best_idx];
  return true;
}

// -----------------------------------------------------------------------------
cv::Mat
map_keypoints_to_camera
::compute_sgbm_disparity(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect )
{
  if( !m_stereo_depth_map_algorithm )
  {
    // Algorithm not configured, return empty
    return cv::Mat();
  }

  // Convert cv::Mat to ImageContainers
  kv::image_container_sptr left_container =
    std::make_shared< kwiver::arrows::ocv::image_container >(
      left_image_rect, kwiver::arrows::ocv::image_container::BGR_COLOR );
  kv::image_container_sptr right_container =
    std::make_shared< kwiver::arrows::ocv::image_container >(
      right_image_rect, kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Compute disparity using the algorithm
  kv::image_container_sptr disparity_container =
    m_stereo_depth_map_algorithm->compute( left_container, right_container );

  if( !disparity_container )
  {
    return cv::Mat();
  }

  // Convert result back to cv::Mat
  return kwiver::arrows::ocv::image_container::vital_to_ocv(
    disparity_container->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_sgbm(
  const cv::Mat& disparity_map,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect ) const
{
  int x = static_cast< int >( left_point_rect.x() + 0.5 );
  int y = static_cast< int >( left_point_rect.y() + 0.5 );

  // Check bounds
  if( x < 0 || x >= disparity_map.cols || y < 0 || y >= disparity_map.rows )
  {
    return false;
  }

  // Get disparity value (SGBM returns fixed-point values scaled by 16)
  short disp_raw = disparity_map.at< short >( y, x );

  // Check for invalid disparity (OpenCV marks invalid as negative values)
  if( disp_raw < 0 )
  {
    return false;
  }

  // Convert to float disparity
  double disparity = static_cast< double >( disp_raw ) / 16.0;

  // Compute right point
  right_point_rect = kv::vector_2d( left_point_rect.x() - disparity, left_point_rect.y() );

  return true;
}

// -----------------------------------------------------------------------------
const cv::Mat&
map_keypoints_to_camera
::get_rectification_map_x( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_x : m_rectification_map_left_x;
}

// -----------------------------------------------------------------------------
const cv::Mat&
map_keypoints_to_camera
::get_rectification_map_y( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_y : m_rectification_map_left_y;
}

#endif // VIAME_ENABLE_OPENCV

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_external_disparity(
  const kv::image_container_sptr& disparity_image,
  const kv::vector_2d& left_point,
  kv::vector_2d& right_point,
  int search_window ) const
{
  if( !disparity_image )
  {
    return false;
  }

  const auto& img = disparity_image->get_image();
  int cx = static_cast< int >( left_point.x() + 0.5 );
  int cy = static_cast< int >( left_point.y() + 0.5 );
  int w = static_cast< int >( img.width() );
  int h = static_cast< int >( img.height() );

  // Check center pixel bounds
  if( cx < 0 || cx >= w || cy < 0 || cy >= h )
  {
    return false;
  }

  // Cast to char* for pointer arithmetic (void* arithmetic is undefined)
  const char* img_data = reinterpret_cast<const char*>( img.first_pixel() );

  // Helper lambda: read disparity at (px, py), returns <= 0 if invalid
  auto read_disparity = [&]( int px, int py ) -> double
  {
    if( img.pixel_traits().type == kv::image_pixel_traits::UNSIGNED &&
        img.pixel_traits().num_bytes == 2 )
    {
      const uint16_t* ptr = reinterpret_cast<const uint16_t*>(
        img_data + py * img.h_step() + px * img.w_step() );
      return static_cast< double >( *ptr ) / 256.0;
    }
    else if( img.pixel_traits().type == kv::image_pixel_traits::SIGNED &&
             img.pixel_traits().num_bytes == 2 )
    {
      const int16_t* ptr = reinterpret_cast<const int16_t*>(
        img_data + py * img.h_step() + px * img.w_step() );
      int16_t raw_val = *ptr;
      if( raw_val < 0 )
      {
        return -1.0;
      }
      return static_cast< double >( raw_val ) / 16.0;
    }
    else if( img.pixel_traits().type == kv::image_pixel_traits::FLOAT &&
             img.pixel_traits().num_bytes == 4 )
    {
      const float* ptr = reinterpret_cast<const float*>(
        img_data + py * img.h_step() + px * img.w_step() );
      return static_cast< double >( *ptr );
    }
    return -1.0;
  };

  double disparity = 0.0;

  if( search_window <= 0 )
  {
    // Original single-pixel lookup
    disparity = read_disparity( cx, cy );

    if( disparity <= 0.0 || !std::isfinite( disparity ) )
    {
      return false;
    }
  }
  else
  {
    // Neighborhood median lookup over (2w+1) x (2w+1) window
    int x_min = std::max( 0, cx - search_window );
    int x_max = std::min( w - 1, cx + search_window );
    int y_min = std::max( 0, cy - search_window );
    int y_max = std::min( h - 1, cy + search_window );

    std::vector< double > valid_disparities;
    valid_disparities.reserve(
      ( x_max - x_min + 1 ) * ( y_max - y_min + 1 ) );

    for( int py = y_min; py <= y_max; ++py )
    {
      for( int px = x_min; px <= x_max; ++px )
      {
        double d = read_disparity( px, py );
        if( d > 0.0 && std::isfinite( d ) )
        {
          valid_disparities.push_back( d );
        }
      }
    }

    if( valid_disparities.empty() )
    {
      return false;
    }

    size_t mid = valid_disparities.size() / 2;
    std::nth_element( valid_disparities.begin(),
                      valid_disparities.begin() + mid,
                      valid_disparities.end() );
    disparity = valid_disparities[ mid ];
  }

  // Compute right point (standard stereo: right_x = left_x - disparity)
  right_point = kv::vector_2d( left_point.x() - disparity, left_point.y() );

  return true;
}

// -----------------------------------------------------------------------------
std::vector< std::string >
parse_matching_methods( const std::string& methods_str )
{
  std::vector< std::string > methods;
  std::stringstream ss( methods_str );
  std::string method;

  while( std::getline( ss, method, ',' ) )
  {
    // Trim whitespace
    size_t start = method.find_first_not_of( " \t" );
    size_t end = method.find_last_not_of( " \t" );

    if( start != std::string::npos && end != std::string::npos )
    {
      methods.push_back( method.substr( start, end - start + 1 ) );
    }
  }

  return methods;
}

// -----------------------------------------------------------------------------
bool
method_requires_images( const std::string& method )
{
  return ( method == "template_matching" ||
           method == "epipolar_template_matching" ||
           method == "feature_descriptor" ||
           method == "ransac_feature" ||
           method == "compute_disparity" );
}

// -----------------------------------------------------------------------------
std::vector< std::string >
get_valid_methods()
{
  return {
    "input_pairs_only",
    "depth_projection",
    "external_disparity",
    "compute_disparity",
    "template_matching",
    "epipolar_template_matching",
    "feature_descriptor",
    "ransac_feature"
  };
}

} // end namespace core

} // end namespace viame
