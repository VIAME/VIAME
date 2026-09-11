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

#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/util/string.h>

#include <viame/measurement/triangulate.h>

#include <viame/measurement/projection.h>

#include <image_ops/color.h>
#include <image_ops/draw.h>
#include <image_ops/layout.h>
#include <image_ops/match.h>
#include <image_ops/resample.h>
#include <image_ops/warp.h>

#include <viame/video_io/codecs/image_codec.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>

// The Python bindings for these utilities live at the bottom of this file under
// VIAME_MEASUREMENT_PYTHON_BINDINGS (defined only for the _measurement module
// build). The implementation below is skipped in that build so the module just
// wraps the symbols exported by the viame_core library it links against.
#ifndef VIAME_MEASUREMENT_PYTHON_BINDINGS

namespace viame
{

namespace core
{

// =============================================================================
// parse_length_from_notes
// =============================================================================

double
parse_length_from_notes( const kv::detected_object_sptr& det )
{
  if( !det )
    return -1.0;

  for( const auto& note : det->notes() )
  {
    if( note.size() > 8 && note.substr( 0, 8 ) == ":length=" )
    {
      try
      {
        return std::stod( note.substr( 8 ) );
      }
      catch( ... )
      {
        return -1.0;
      }
    }
  }
  return -1.0;
}

// =============================================================================
// parse_stereo_rms_from_notes
// =============================================================================

double
parse_stereo_rms_from_notes( const kv::detected_object_sptr& det )
{
  if( !det )
    return -1.0;

  static const std::string prefix = ":stereo_rms=";
  for( const auto& note : det->notes() )
  {
    if( note.size() > prefix.size() &&
        note.compare( 0, prefix.size(), prefix ) == 0 )
    {
      try
      {
        return std::stod( note.substr( prefix.size() ) );
      }
      catch( ... )
      {
        return -1.0;
      }
    }
  }
  return -1.0;
}

// =============================================================================
// DINOv3 Python C API helpers
// =============================================================================

namespace
{

static auto logger = kwiver::vital::get_logger( "viame.core.measurement_utilities" );

namespace io = viame::image_ops;

/// The greyscale of a colour image.
///
/// `cv::cvtColor`'s `BGR2GRAY` on the BGR mat the bridge used to build,
/// which is `rgb_to_gray` on the RGB image that mat came from: the same
/// BT.601 weights on the same channels. An image that is already one plane
/// comes back unchanged, and a fourth plane is dropped, as `BGRA2GRAY` did.
kv::image_of< uint8_t >
to_gray( kv::image_of< uint8_t > const& image )
{
  if( image.depth() == 1 )
  {
    return image;
  }

  return io::rgb_to_gray( image );
}

/// Three planes whatever came in: one plane is replicated, four are cut
/// down. `cv::cvtColor`'s `GRAY2BGR` and `BGRA2BGR`, on RGB.
kv::image_of< uint8_t >
to_three_planes( kv::image_of< uint8_t > const& image )
{
  if( image.depth() == 1 )
  {
    return io::gray_to_rgb( image );
  }

  if( image.depth() == 3 )
  {
    return image;
  }

  kv::image_of< uint8_t > out( image.width(), image.height(), 3 );

  for( size_t plane = 0; plane < 3; ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        out( i, j, plane ) = image( i, j, plane );
      }
    }
  }

  return out;
}

/// A region of an image, which was `image( cv::Rect( ... ) )`.
///
/// `io::crop` copies where OpenCV aliased. Every caller here either reads
/// the region or hands it to a matcher, so the copy costs a patch and buys
/// not having to think about lifetimes.
kv::image_of< uint8_t >
region( kv::image_of< uint8_t > const& image, image_rect const& rect )
{
  return io::crop( image, static_cast< size_t >( rect.x ),
                   static_cast< size_t >( rect.y ),
                   static_cast< size_t >( rect.width ),
                   static_cast< size_t >( rect.height ) );
}

} // end anonymous namespace (logger)

#ifdef VIAME_ENABLE_PYTHON

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

/// The bytes a `numpy` array of this image would hold: row major, channels
/// interleaved, which is what `cv::Mat::data` was and what the python side
/// reads. `vital::image` is planar, so this is the one place the two layouts
/// have to be spelled out rather than aliased.
std::vector< uint8_t >
interleaved( const kv::image_of< uint8_t >& image )
{
  std::vector< uint8_t > out;
  out.reserve( image.width() * image.height() * image.depth() );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t plane = 0; plane < image.depth(); ++plane )
      {
        out.push_back( image( i, j, plane ) );
      }
    }
  }

  return out;
}

/// Call set_images_from_bytes to load a new stereo image pair
bool dino_set_images( const kv::image_of< uint8_t >& left,
                      const kv::image_of< uint8_t >& right )
{
  if( !s_dino_module )
  {
    return false;
  }

  python_gil_guard gil;

  auto const left_cont = interleaved( left );
  auto const right_cont = interleaved( right );

  // Create Python bytes objects wrapping the image data
  PyObject* left_bytes = PyBytes_FromStringAndSize(
    reinterpret_cast< const char* >( left_cont.data() ),
    static_cast< Py_ssize_t >( left_cont.size() ) );

  PyObject* right_bytes = PyBytes_FromStringAndSize(
    reinterpret_cast< const char* >( right_cont.data() ),
    static_cast< Py_ssize_t >( right_cont.size() ) );

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
    left_bytes, static_cast< int >( left.height() ),
    static_cast< int >( left.width() ), static_cast< int >( left.depth() ),
    right_bytes, static_cast< int >( right.height() ),
    static_cast< int >( right.width() ),
    static_cast< int >( right.depth() ) );

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

#endif // VIAME_ENABLE_PYTHON

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
  , uniqueness_ratio( 0.85 )
  , record_stereo_method( true )
  , refine_keypoints_with_disparity( false )
  , refine_keypoints_disparity_window( 7 )
  , refine_keypoints_reject_inconsistent( false )
  , refine_keypoints_max_distance( 0.25 )
  , debug_epipolar_directory( "" )
  , detection_pairing_method( "" )
  , detection_pairing_threshold( 0.1 )
  , detection_pairing_require_class_match( true )
  , detection_pairing_use_optimal_assignment( true )
  , dino_crop_max_area_ratio( 0.05 )
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

  config->set_value( "uniqueness_ratio", uniqueness_ratio,
    "Uniqueness ratio for epipolar NCC template matching (Lowe's ratio test). "
    "After finding the best match score, compares it to the second-best match. "
    "If second_best / best > this ratio, the match is considered ambiguous and "
    "is rejected. This prevents false matches on repetitive textures (e.g. fish "
    "scales) where multiple locations score similarly. "
    "Set to 0 to disable. Default is 0.85. "
    "Lower values are more strict (reject more ambiguous matches)." );

  config->set_value( "record_stereo_method", record_stereo_method,
    "If true, record the stereo measurement method used as an attribute on each "
    "output detection object. The attribute will be ':stereo_method=METHOD' "
    "where METHOD is one of: input_kps_used, input_kps_disparity_refined, "
    "input_kps_partial_disparity_refined, disparity_inconsistent_rejected, "
    "template_matching, epipolar_template_matching, feature_descriptor, "
    "ransac_feature, depth_projection, external_disparity, or "
    "compute_disparity." );

  config->set_value( "refine_keypoints_with_disparity", refine_keypoints_with_disparity,
    "If true and a stereo_disparity algorithm is configured, snap right "
    "keypoints of already-paired tracks to the disparity-implied match of "
    "their left counterparts. Falls back to the original right keypoint "
    "when disparity is invalid at the query location. Tracks that lacked "
    "a right keypoint are unaffected." );

  config->set_value( "refine_keypoints_disparity_window", refine_keypoints_disparity_window,
    "Half-width (in pixels) of the neighborhood sampled when reading the "
    "disparity map for keypoint refinement (median over (2w+1)^2). Set to "
    "0 for single-pixel lookup. Only applies when "
    "refine_keypoints_with_disparity is true." );

  config->set_value( "refine_keypoints_reject_inconsistent", refine_keypoints_reject_inconsistent,
    "If true, compare each tracker-provided right keypoint to its "
    "disparity-implied position; if the distance exceeds "
    "refine_keypoints_max_distance (normalized by L bbox size), reject the "
    "track's measurement instead of refining. Only applies when "
    "refine_keypoints_with_disparity is true." );

  config->set_value( "refine_keypoints_max_distance", refine_keypoints_max_distance,
    "Maximum allowed distance between tracker and disparity-implied right "
    "keypoint, as a fraction of the left bbox max(width,height). Above this "
    "threshold the track is rejected when "
    "refine_keypoints_reject_inconsistent is true. Set to 0 to disable." );

  config->set_value( "debug_epipolar_directory", debug_epipolar_directory,
    "Directory to write debug images showing epipolar search lines overlaid on "
    "the source and target images. Each keypoint match attempt writes a side-by-side "
    "image with the source point marked on the left image and the bounded epipolar "
    "curve drawn on the right image, along with any matched point. "
    "Set to empty string (default) to disable debug output." );

  config->set_value( "detection_pairing_method", detection_pairing_method,
    "Method for pairing left/right detections that do not share the same track ID. "
    "Set to empty string (default) to disable detection pairing. "
    "Valid options: 'iou' (bounding box overlap), "
    "'calibration' (stereo reprojection error), "
    "'feature_matching' (visual feature detection and matching within bounding boxes), "
    "'epipolar_iou' (project left bbox to right using depth, match by IOU), "
    "'keypoint_projection' (project left head/tail keypoints to right, match by pixel distance)." );

  config->set_value( "detection_pairing_threshold", detection_pairing_threshold,
    "Threshold for detection pairing. For 'iou'/'epipolar_iou' this is the minimum IOU "
    "(default 0.1). For 'calibration' this is the max reprojection error in pixels. "
    "For 'keypoint_projection' this is the max average keypoint pixel distance." );

  config->set_value( "detection_pairing_require_class_match", detection_pairing_require_class_match,
    "If true, only pair detections whose top class labels match (default true)." );

  config->set_value( "detection_pairing_use_optimal_assignment", detection_pairing_use_optimal_assignment,
    "If true, use greedy optimal assignment to maximize matching quality. "
    "If false, use simple sequential matching (default true)." );

  config->set_value( "dino_crop_max_area_ratio", dino_crop_max_area_ratio,
    "Maximum fraction of full image area that the union of left and right DINO "
    "crops may occupy. When all epipolar regions for a frame fit within this "
    "fraction, DINO runs on cropped subimages instead of the full resolution, "
    "proportionally reducing ViT inference cost. Set to 0 to disable cropping. "
    "Default 0.05 (5%). Empirically, crops above ~5% degrade DINO accuracy "
    "because medium-sized crops lose global context without sufficiently "
    "reducing the candidate space. Keep this low for robustness." );

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
  kv::get_nested_algo_configuration<kv::algo::detect_features>(
    "feature_detector", config, feature_detector );
  kv::get_nested_algo_configuration<kv::algo::extract_descriptors>(
    "descriptor_extractor", config, descriptor_extractor );
  kv::get_nested_algo_configuration<kv::algo::match_features>(
    "feature_matcher", config, feature_matcher );
  kv::get_nested_algo_configuration<kv::algo::estimate_fundamental_matrix>(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );
  kv::get_nested_algo_configuration<kv::algo::compute_stereo_depth_map>(
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
  uniqueness_ratio = config->get_value< double >( "uniqueness_ratio", uniqueness_ratio );
  record_stereo_method = config->get_value< bool >( "record_stereo_method", record_stereo_method );
  refine_keypoints_with_disparity = config->get_value< bool >( "refine_keypoints_with_disparity", refine_keypoints_with_disparity );
  refine_keypoints_disparity_window = config->get_value< int >( "refine_keypoints_disparity_window", refine_keypoints_disparity_window );
  refine_keypoints_reject_inconsistent = config->get_value< bool >( "refine_keypoints_reject_inconsistent", refine_keypoints_reject_inconsistent );
  refine_keypoints_max_distance = config->get_value< double >( "refine_keypoints_max_distance", refine_keypoints_max_distance );
  debug_epipolar_directory = config->get_value< std::string >( "debug_epipolar_directory", debug_epipolar_directory );
  detection_pairing_method = config->get_value< std::string >( "detection_pairing_method", detection_pairing_method );
  detection_pairing_threshold = config->get_value< double >( "detection_pairing_threshold", detection_pairing_threshold );
  detection_pairing_require_class_match = config->get_value< bool >( "detection_pairing_require_class_match", detection_pairing_require_class_match );
  detection_pairing_use_optimal_assignment = config->get_value< bool >( "detection_pairing_use_optimal_assignment", detection_pairing_use_optimal_assignment );
  dino_crop_max_area_ratio = config->get_value< double >( "dino_crop_max_area_ratio", dino_crop_max_area_ratio );
  dino_model_name = config->get_value< std::string >( "dino_model_name", dino_model_name );
  dino_threshold = config->get_value< double >( "dino_threshold", dino_threshold );
  dino_weights_path = config->get_value< std::string >( "dino_weights_path", dino_weights_path );
  dino_top_k = config->get_value< int >( "dino_top_k", dino_top_k );

  // Configure nested algorithms
  kv::set_nested_algo_configuration<kv::algo::detect_features>(
    "feature_detector", config, feature_detector );
  kv::set_nested_algo_configuration<kv::algo::extract_descriptors>(
    "descriptor_extractor", config, descriptor_extractor );
  kv::set_nested_algo_configuration<kv::algo::match_features>(
    "feature_matcher", config, feature_matcher );
  kv::set_nested_algo_configuration<kv::algo::estimate_fundamental_matrix>(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );
  kv::set_nested_algo_configuration<kv::algo::compute_stereo_depth_map>(
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
    valid = kv::check_nested_algo_configuration<kv::algo::detect_features>(
      "feature_detector", config ) && valid;
  }
  if( config->has_value( "descriptor_extractor:type" ) &&
      config->get_value< std::string >( "descriptor_extractor:type" ) != "" )
  {
    valid = kv::check_nested_algo_configuration<kv::algo::extract_descriptors>(
      "descriptor_extractor", config ) && valid;
  }
  if( config->has_value( "feature_matcher:type" ) &&
      config->get_value< std::string >( "feature_matcher:type" ) != "" )
  {
    valid = kv::check_nested_algo_configuration<kv::algo::match_features>(
      "feature_matcher", config ) && valid;
  }
  if( config->has_value( "fundamental_matrix_estimator:type" ) &&
      config->get_value< std::string >( "fundamental_matrix_estimator:type" ) != "" )
  {
    valid = kv::check_nested_algo_configuration<kv::algo::estimate_fundamental_matrix>(
      "fundamental_matrix_estimator", config ) && valid;
  }
  if( config->has_value( "stereo_disparity:type" ) &&
      config->get_value< std::string >( "stereo_disparity:type" ) != "" )
  {
    valid = kv::check_nested_algo_configuration<kv::algo::compute_stereo_depth_map>(
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
  , m_uniqueness_ratio( 0.85 )
  , m_debug_epipolar_directory( "" )
  , m_debug_frame_counter( 0 )
  , m_dino_model_name( "dinov2_vitb14" )
  , m_dino_threshold( 0.0 )
  , m_dino_weights_path( "" )
  , m_dino_top_k( 100 )
  , m_dino_crop_max_area_ratio( 0.05 )
  , m_cached_frame_id( -1 )
  , m_dino_full_images_set( false )
  , m_dino_crop_active( false )
  , m_rectification_computed( false )
  , m_rectification_valid( false )
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
::set_dino_params( const std::string& model_name, double threshold,
                   const std::string& weights_path, int top_k,
                   double crop_max_area_ratio )
{
  m_dino_model_name = model_name;
  m_dino_threshold = threshold;
  m_dino_weights_path = weights_path;
  m_dino_top_k = top_k;
  m_dino_crop_max_area_ratio = crop_max_area_ratio;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_epipolar_descriptor_type( const std::string& descriptor_type )
{
  m_epipolar_descriptor_type = descriptor_type;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::set_uniqueness_ratio( double ratio )
{
  m_uniqueness_ratio = ratio;
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

  m_uniqueness_ratio = settings.uniqueness_ratio;
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
bool
map_keypoints_to_camera
::find_corresponding_point_epipolar(
  const kv::image_of< uint8_t >& source_colour,
  const kv::image_of< uint8_t >& target_colour,
  const kv::vector_2d& source_point,
  const std::vector< kv::vector_2d >& epipolar_points,
  kv::vector_2d& target_point )
{
  if( epipolar_points.empty() )
    return false;

  if( m_epipolar_descriptor_type == "ncc" )
  {
    auto const source_gray = to_gray( source_colour );
    auto const target_gray = to_gray( target_colour );

    auto t_ncc_start = std::chrono::steady_clock::now();

    bool found = find_corresponding_point_epipolar_template_matching(
      source_gray, target_gray, source_point, epipolar_points, target_point );

    auto t_ncc_end = std::chrono::steady_clock::now();
    LOG_INFO( logger, "NCC point-by-point ("
      << epipolar_points.size() << " pts): "
      << std::chrono::duration_cast< std::chrono::milliseconds >(
           t_ncc_end - t_ncc_start ).count() << "ms" );

    return found;
  }
  else if( m_epipolar_descriptor_type == "ncc_strip" )
  {
    auto const source_gray = to_gray( source_colour );
    auto const target_gray = to_gray( target_colour );

    auto t_strip_start = std::chrono::steady_clock::now();

    bool found = find_corresponding_point_epipolar_strip_ncc(
      source_gray, target_gray, source_point, epipolar_points, target_point );

    auto t_strip_end = std::chrono::steady_clock::now();
    LOG_INFO( logger, "Strip NCC (1 kp): "
      << std::chrono::duration_cast< std::chrono::milliseconds >(
           t_strip_end - t_strip_start ).count() << "ms" );

    return found;
  }
#ifdef VIAME_ENABLE_PYTHON
  else if( m_epipolar_descriptor_type == "dino" )
  {
    if( !dino_ensure_initialized(
          m_dino_model_name, m_dino_threshold, m_dino_weights_path ) )
    {
      throw std::runtime_error(
        "DINO matcher failed to initialize. "
        "Ensure viame.pytorch.dino_matcher is installed and PyTorch is available." );
    }

    int dino_img_w = static_cast< int >( source_colour.width() );
    int dino_img_h = static_cast< int >( source_colour.height() );
    int dino_right_img_w = static_cast< int >( target_colour.width() );
    int dino_right_img_h = static_cast< int >( target_colour.height() );

    // Prepare grayscale images for NCC refinement (when using top-K)
    kv::image_of< uint8_t > source_gray, target_gray;
    if( m_dino_top_k > 0 )
    {
      source_gray = to_gray( source_colour );
      target_gray = to_gray( target_colour );
    }

    int crop_pad = m_template_size / 2 + 16;
    int dino_patch_size = 14;

    // Compute per-keypoint crop regions
    bool kp_crop_active = false;
    image_rect kp_left_crop, kp_right_crop;

    if( m_dino_crop_max_area_ratio > 0.0 )
    {
      double full_area = static_cast< double >( dino_img_w ) * dino_img_h;

      // Epipolar bounding box for this keypoint
      double epi_min_x = epipolar_points[0].x(), epi_max_x = epipolar_points[0].x();
      double epi_min_y = epipolar_points[0].y(), epi_max_y = epipolar_points[0].y();
      for( const auto& ep : epipolar_points )
      {
        epi_min_x = std::min( epi_min_x, ep.x() );
        epi_max_x = std::max( epi_max_x, ep.x() );
        epi_min_y = std::min( epi_min_y, ep.y() );
        epi_max_y = std::max( epi_max_y, ep.y() );
      }

      double epi_area = ( epi_max_x - epi_min_x ) * ( epi_max_y - epi_min_y );

      if( full_area > 0 && epi_area / full_area <= m_dino_crop_max_area_ratio )
      {
        // Left crop: small region around the source keypoint
        int lx0 = std::max( 0, static_cast< int >( source_point.x() ) - crop_pad );
        int ly0 = std::max( 0, static_cast< int >( source_point.y() ) - crop_pad );
        int lx1 = std::min( dino_img_w, static_cast< int >( source_point.x() ) + crop_pad );
        int ly1 = std::min( dino_img_h, static_cast< int >( source_point.y() ) + crop_pad );

        // Align to DINO patch size
        int lw = ( ( lx1 - lx0 + dino_patch_size - 1 ) / dino_patch_size ) * dino_patch_size;
        int lh = ( ( ly1 - ly0 + dino_patch_size - 1 ) / dino_patch_size ) * dino_patch_size;
        lx1 = std::min( dino_img_w, lx0 + lw );
        ly1 = std::min( dino_img_h, ly0 + lh );

        // Right crop: epipolar bounding box with padding
        int rx0 = std::max( 0, static_cast< int >( std::floor( epi_min_x ) ) - crop_pad );
        int ry0 = std::max( 0, static_cast< int >( std::floor( epi_min_y ) ) - crop_pad );
        int rx1 = std::min( dino_right_img_w, static_cast< int >( std::ceil( epi_max_x ) ) + crop_pad );
        int ry1 = std::min( dino_right_img_h, static_cast< int >( std::ceil( epi_max_y ) ) + crop_pad );

        int rw = ( ( rx1 - rx0 + dino_patch_size - 1 ) / dino_patch_size ) * dino_patch_size;
        int rh = ( ( ry1 - ry0 + dino_patch_size - 1 ) / dino_patch_size ) * dino_patch_size;
        rx1 = std::min( dino_right_img_w, rx0 + rw );
        ry1 = std::min( dino_right_img_h, ry0 + rh );

        kp_left_crop = image_rect( lx0, ly0, lx1 - lx0, ly1 - ly0 );
        kp_right_crop = image_rect( rx0, ry0, rx1 - rx0, ry1 - ry0 );
        kp_crop_active = true;
      }
    }

    // Skip DINO extraction when using full images that are already cached
    // for this frame (across keypoints and detections)
    if( kp_crop_active || !m_dino_full_images_set )
    {
      auto const dino_left = kp_crop_active
        ? region( source_colour, kp_left_crop ) : source_colour;
      auto const dino_right = kp_crop_active
        ? region( target_colour, kp_right_crop ) : target_colour;

      auto t_dino_start = std::chrono::steady_clock::now();
      bool ok = dino_set_images( dino_left, dino_right );
      auto t_dino_end = std::chrono::steady_clock::now();

      LOG_INFO( logger, "DINO extraction: "
        << std::chrono::duration_cast< std::chrono::milliseconds >(
             t_dino_end - t_dino_start ).count() << "ms (L="
        << dino_left.width() << "x" << dino_left.height()
        << " R=" << dino_right.width() << "x" << dino_right.height()
        << ( kp_crop_active ? " per-kp crop" : " full" ) << ")" );

      if( !ok )
      {
        LOG_WARN( logger, "DINO set_images failed (L="
          << dino_left.width() << "x" << dino_left.height()
          << " R=" << dino_right.width() << "x" << dino_right.height()
          << ( kp_crop_active ? " per-kp crop" : " full" )
          << "). Skipping this keypoint." );
        return false;
      }

      if( !kp_crop_active )
        m_dino_full_images_set = true;
    }

    double kp_left_off_x = kp_crop_active ? kp_left_crop.x : 0;
    double kp_left_off_y = kp_crop_active ? kp_left_crop.y : 0;
    double kp_right_off_x = kp_crop_active ? kp_right_crop.x : 0;
    double kp_right_off_y = kp_crop_active ? kp_right_crop.y : 0;

    if( m_dino_top_k > 0 )
    {
      // Offset epipolar points for DINO crop coordinates
      std::vector< kv::vector_2d > dino_epi;
      dino_epi.reserve( epipolar_points.size() );
      for( const auto& pt : epipolar_points )
        dino_epi.push_back( kv::vector_2d(
          pt.x() - kp_right_off_x, pt.y() - kp_right_off_y ) );

      double dino_src_x = source_point.x() - kp_left_off_x;
      double dino_src_y = source_point.y() - kp_left_off_y;

      auto indices = dino_get_top_k_indices(
        dino_src_x, dino_src_y, dino_epi, m_dino_top_k );

      LOG_INFO( logger, "DINO top-K: src=(" << source_point.x() << "," << source_point.y()
        << ") crop_src=(" << dino_src_x << "," << dino_src_y
        << ") epi_pts=" << dino_epi.size()
        << " top_k=" << indices.size()
        << " left_img=" << source_colour.width() << "x" << source_colour.height()
        << " right_img=" << target_colour.width() << "x" << target_colour.height()
        << " cached=" << ( m_dino_full_images_set ? "yes" : "no" ) );

      if( indices.empty() )
        return false;

      std::vector< kv::vector_2d > filtered;
      filtered.reserve( indices.size() );
      for( int idx : indices )
        filtered.push_back( epipolar_points[idx] );

      // NCC refinement on full-resolution grayscale images
      bool ncc_ok = find_corresponding_point_epipolar_template_matching(
        source_gray, target_gray, source_point, filtered, target_point );

      LOG_INFO( logger, "NCC result: " << ( ncc_ok ? "MATCHED" : "REJECTED" )
        << " src=(" << source_point.x() << "," << source_point.y() << ")"
        << " gray_planes=" << source_gray.depth()
        << " gray_size=" << source_gray.width() << "x" << source_gray.height() );

      return ncc_ok;
    }
    else
    {
      // DINO-only mode
      std::vector< kv::vector_2d > dino_epi;
      dino_epi.reserve( epipolar_points.size() );
      for( const auto& pt : epipolar_points )
        dino_epi.push_back( kv::vector_2d(
          pt.x() - kp_right_off_x, pt.y() - kp_right_off_y ) );

      auto match = dino_match_point(
        source_point.x() - kp_left_off_x, source_point.y() - kp_left_off_y,
        dino_epi, m_dino_threshold );

      if( match.success )
        target_point = kv::vector_2d(
          match.x + kp_right_off_x, match.y + kp_right_off_y );
      return match.success;
    }
  }
#endif // VIAME_ENABLE_PYTHON

  LOG_WARN( logger, "Unknown epipolar descriptor type: " << m_epipolar_descriptor_type );
  return false;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::clear_dino_crop_info()
{
  m_dino_crop_active = false;
  m_dino_left_cropped = kv::image_of< uint8_t >();
  m_dino_right_cropped = kv::image_of< uint8_t >();
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
                         int iw, int ih ) -> image_rect
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

    return image_rect( x0, y0, x1 - x0, y1 - y0 );
  };

  image_rect left_crop = align_crop( left_min_x, left_min_y, left_max_x, left_max_y, img_w, img_h );

  int right_img_w = static_cast< int >( right_image->width() );
  int right_img_h = static_cast< int >( right_image->height() );
  image_rect right_crop = align_crop( right_min_x, right_min_y, right_max_x, right_max_y,
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
  kv::image_of< uint8_t > const left_full( left_image->get_image() );
  kv::image_of< uint8_t > const right_full( right_image->get_image() );

  m_dino_left_crop = left_crop;
  m_dino_right_crop = right_crop;
  m_dino_left_cropped = region( left_full, left_crop );
  m_dino_right_cropped = region( right_full, right_crop );
  m_dino_crop_active = true;

  LOG_INFO( logger, "DINO crop: left " << left_crop.width << "x" << left_crop.height
    << " right " << right_crop.width << "x" << right_crop.height
    << " (avg ratio " << avg_ratio << ")" );
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
  det->add_note( ":length=" + std::to_string( measurement.length ) );
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
  kv::vector_< 2, double > left_pt( left_point.x(), left_point.y() );
  kv::vector_< 2, double > right_pt( right_point.x(), right_point.y() );

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
double
aggregate_lengths(
  const std::vector< double >& input_lengths,
  const std::string& method,
  double iqr_factor )
{
  // Keep only valid (positive) lengths
  std::vector< double > lengths;
  lengths.reserve( input_lengths.size() );
  for( double v : input_lengths )
  {
    if( v > 0.0 )
    {
      lengths.push_back( v );
    }
  }

  if( lengths.empty() )
  {
    return -1.0;
  }

  std::sort( lengths.begin(), lengths.end() );

  if( method == "average_iqr" )
  {
    size_t n = lengths.size();
    double q1 = lengths[ n / 4 ];
    double q3 = lengths[ ( 3 * n ) / 4 ];
    double iqr = q3 - q1;
    double lower = q1 - iqr_factor * iqr;
    double upper = q3 + iqr_factor * iqr;

    double sum = 0.0;
    int count = 0;
    for( double v : lengths )
    {
      if( v >= lower && v <= upper )
      {
        sum += v;
        ++count;
      }
    }

    if( count == 0 )
    {
      return -1.0;
    }

    return sum / count;
  }
  else if( method == "median" )
  {
    size_t m = lengths.size();
    if( m % 2 == 0 )
    {
      return ( lengths[ m / 2 - 1 ] + lengths[ m / 2 ] ) / 2.0;
    }
    return lengths[ m / 2 ];
  }

  // "average" (default): plain mean
  double sum = 0.0;
  for( double v : lengths )
  {
    sum += v;
  }
  return sum / lengths.size();
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

  // Prepare stereo images if needed
  bool has_images = ( left_image && right_image );
  if( has_images )
  {
    m_cached_stereo_images = prepare_stereo_images(
      methods, left_cam, right_cam, left_image, right_image );
  }

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
    else if( method == "compute_disparity" && m_stereo_depth_map_algorithm &&
             m_cached_stereo_images.rectified_available )
    {
      // Compute disparity map using the configured algorithm if not cached for this frame
      if( !m_cached_compute_disparity )
      {
        kv::image_container_sptr left_rect_container =
          std::make_shared< kv::simple_image_container >(
            m_cached_stereo_images.left_rectified );
        kv::image_container_sptr right_rect_container =
          std::make_shared< kv::simple_image_container >(
            m_cached_stereo_images.right_rectified );

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
      auto const disp_hint = m_cached_stereo_images.disparity_available
        ? m_cached_stereo_images.disparity_map : nullptr;

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

      kv::image_of< uint8_t > const left_bgr( left_image->get_image() );
      kv::image_of< uint8_t > const right_bgr( right_image->get_image() );

      head_found = find_corresponding_point_epipolar(
        left_bgr, right_bgr, result.left_head, epipolar_head, result.right_head );
      tail_found = find_corresponding_point_epipolar(
        left_bgr, right_bgr, result.left_tail, epipolar_tail, result.right_tail );

      bool descriptor_available = true;
      std::string descriptor_label = m_epipolar_descriptor_type;

      // Debug: write images with epipolar curves overlaid
      if( descriptor_available && !m_debug_epipolar_directory.empty() )
      {
        // Three planes whatever came in, so the overlay colours below mean
        // what they say. RGB rather than the BGR the bridge used to hand
        // back, so the triples are written in that order.
        auto const left_color = to_three_planes(
          kv::image_of< uint8_t >( left_image->get_image() ) );
        auto const right_color = to_three_planes(
          kv::image_of< uint8_t >( right_image->get_image() ) );

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

          auto left_draw = left_color;
          auto right_draw = right_color;

          long const src_x = static_cast< long >( kp.src_pt.x() + 0.5 );
          long const src_y = static_cast< long >( kp.src_pt.y() + 0.5 );

          io::colour const cyan{ 0, 255, 255 };
          io::colour const green{ 0, 255, 0 };
          io::colour const orange{ 255, 200, 0 };
          io::colour const magenta{ 200, 0, 255 };
          io::colour const red{ 255, 0, 0 };

          io::draw_circle( left_draw, src_x, src_y, 8, cyan, 2 );
          io::draw_line( left_draw, src_x - 12, src_y, src_x + 12, src_y, cyan );
          io::draw_line( left_draw, src_x, src_y - 12, src_x, src_y + 12, cyan );

          if( kp.epi_pts.size() >= 2 )
          {
            // `cv::polylines` over the sampled curve, which is a run of
            // segments between consecutive samples.
            for( size_t i = 1; i < kp.epi_pts.size(); ++i )
            {
              io::draw_line(
                right_draw,
                static_cast< long >( kp.epi_pts[ i - 1 ].x() + 0.5 ),
                static_cast< long >( kp.epi_pts[ i - 1 ].y() + 0.5 ),
                static_cast< long >( kp.epi_pts[ i ].x() + 0.5 ),
                static_cast< long >( kp.epi_pts[ i ].y() + 0.5 ), green, 2 );
            }

            io::draw_circle(
              right_draw,
              static_cast< long >( kp.epi_pts.front().x() + 0.5 ),
              static_cast< long >( kp.epi_pts.front().y() + 0.5 ), 6,
              orange, 2 );
            io::draw_circle(
              right_draw,
              static_cast< long >( kp.epi_pts.back().x() + 0.5 ),
              static_cast< long >( kp.epi_pts.back().y() + 0.5 ), 6,
              magenta, 2 );
          }

          if( kp.found )
          {
            long const match_x = static_cast< long >( kp.match_pt.x() + 0.5 );
            long const match_y = static_cast< long >( kp.match_pt.y() + 0.5 );

            io::draw_circle( right_draw, match_x, match_y, 8, red, 2 );
            io::draw_line( right_draw, match_x - 12, match_y,
                           match_x + 12, match_y, red );
            io::draw_line( right_draw, match_x, match_y - 12,
                           match_x, match_y + 12, red );
          }

          auto canvas = io::horizontal_concat< uint8_t >(
            { left_draw, right_draw } );

          std::string status = kp.found ? "MATCHED" : "NO MATCH";
          std::string label = descriptor_label + " " + kp.label + " - " + status +
            " (" + std::to_string( kp.epi_pts.size() ) + " samples)";

          // The bitmap font of `image_ops`, not Hershey's, so the glyphs
          // differ; this is a debug overlay nothing is held to.
          io::draw_text( canvas, label, 10, 24, cyan, 2 );

          // PNG rather than JPEG: the writer is `library/video_io`'s now and
          // a lossless overlay is what a person looking at one wants.
          std::string filename = m_debug_epipolar_directory + "/epipolar_" +
            std::to_string( m_debug_frame_counter ) + "_" + kp.label + ".png";
          viame::codecs::write( filename, kv::image( canvas ) );
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

  // Grey, each image independently since they may have different plane
  // counts
  auto const left_grey = to_gray( kv::image_of< uint8_t >(
    left_image->get_image() ) );
  auto const right_grey = to_gray( kv::image_of< uint8_t >(
    right_image->get_image() ) );

  // Compute rectification maps if needed
  compute_rectification_maps( left_cam, right_cam, left_grey.width(),
                              left_grey.height() );

  // Rectify images
  data.left_rectified = rectify_image( left_grey, false );
  data.right_rectified = rectify_image( right_grey, true );
  data.rectified_available = true;

  // Compute disparity if needed for template matching disparity hint
  if( m_use_disparity_hint && m_stereo_depth_map_algorithm )
  {
    data.disparity_map = compute_sgbm_disparity( data.left_rectified,
                                                 data.right_rectified );
    data.disparity_available = ( data.disparity_map != nullptr );
  }

  return data;
}

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
  m_dino_full_images_set = false;
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
::compute_disparity_for_frame(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image )
{
  if( m_cached_compute_disparity )
  {
    return m_cached_compute_disparity;
  }
  if( !m_stereo_depth_map_algorithm || !left_image || !right_image )
  {
    return nullptr;
  }

  // prepare_stereo_images already sets up rectification when given a
  // method list that requests it; "compute_disparity" is the canonical
  // trigger, mirroring how find_stereo_correspondence prepares the data.
  m_cached_stereo_images = prepare_stereo_images(
    { "compute_disparity" }, left_cam, right_cam, left_image, right_image );

  if( !m_cached_stereo_images.rectified_available )
  {
    return nullptr;
  }

  kv::image_container_sptr left_rect_container =
    std::make_shared< kv::simple_image_container >(
      m_cached_stereo_images.left_rectified );
  kv::image_container_sptr right_rect_container =
    std::make_shared< kv::simple_image_container >(
      m_cached_stereo_images.right_rectified );

  m_cached_compute_disparity = m_stereo_depth_map_algorithm->compute(
    left_rect_container, right_rect_container );

  return m_cached_compute_disparity;
}

// -----------------------------------------------------------------------------
kv::vector_2d
map_keypoints_to_camera
::refine_right_point_with_disparity(
  const kv::image_container_sptr& disparity_map,
  const kv::vector_2d& left_point,
  const kv::vector_2d& original_right_point,
  const kv::simple_camera_perspective& right_cam,
  int search_window,
  bool* refined ) const
{
  if( refined ) { *refined = false; }

  if( !disparity_map )
  {
    return original_right_point;
  }

  if( !m_rectification_computed )
  {
    return original_right_point;
  }

  const kv::vector_2d left_rect = rectify_point( left_point, false );

  kv::vector_2d right_rect;
  const bool ok = find_corresponding_point_external_disparity(
    disparity_map, left_rect, right_rect, search_window );

  if( !ok )
  {
    return original_right_point;
  }

  const kv::vector_2d right_unrect =
    unrectify_point( right_rect, true, right_cam );

  if( refined ) { *refined = true; }
  return right_unrect;
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
map_keypoints_to_camera
::get_cached_rectified_left() const
{
  if( m_cached_stereo_images.rectified_available &&
      m_cached_stereo_images.left_rectified.size() > 0 )
  {
    return std::make_shared< kv::simple_image_container >(
      m_cached_stereo_images.left_rectified );
  }

  return nullptr;
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
map_keypoints_to_camera
::get_cached_rectified_right() const
{
  if( m_cached_stereo_images.rectified_available &&
      m_cached_stereo_images.right_rectified.size() > 0 )
  {
    return std::make_shared< kv::simple_image_container >(
      m_cached_stereo_images.right_rectified );
  }

  return nullptr;
}

// -----------------------------------------------------------------------------
void
map_keypoints_to_camera
::compute_rectification_maps(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  size_t width, size_t height )
{
  if( m_rectification_computed )
  {
    return;
  }

  // Get camera intrinsics
  auto left_intrinsics = left_cam.get_intrinsics();
  auto right_intrinsics = right_cam.get_intrinsics();

  m_K1 = left_intrinsics->as_matrix();
  m_K2 = right_intrinsics->as_matrix();

  m_D1.assign( 5, 0.0 );
  m_D2.assign( 5, 0.0 );

  if( m_use_distortion )
  {
    std::vector< double > left_dist = left_intrinsics->dist_coeffs();
    std::vector< double > right_dist = right_intrinsics->dist_coeffs();

    for( size_t i = 0; i < std::min( left_dist.size(), size_t( 5 ) ); ++i )
    {
      m_D1[ i ] = left_dist[ i ];
    }

    for( size_t i = 0; i < std::min( right_dist.size(), size_t( 5 ) ); ++i )
    {
      m_D2[ i ] = right_dist[ i ];
    }
  }

  // Compute rotation and translation from left camera frame to right camera frame
  // X_right = R_relative * X_left + t_relative
  kv::matrix_3x3d R_left = left_cam.rotation().matrix();
  kv::matrix_3x3d R_right = right_cam.rotation().matrix();
  kv::matrix_3x3d R_relative = R_right * R_left.transpose();

  // Translation: t = R_right * (C_left - C_right)
  kv::vector_3d t_relative = R_right * ( left_cam.center() - right_cam.center() );

  // Compute rectification transforms
  auto const rectified = viame::measurement::stereo_rectify(
    m_K1, m_D1, m_K2, m_D2, width, height, R_relative, t_relative );

  m_R1 = rectified.left_rotation;
  m_R2 = rectified.right_rotation;
  m_P1 = rectified.left_projection;
  m_P2 = rectified.right_projection;

  // Compute rectification maps
  viame::measurement::rectification_maps(
    m_K1, m_D1, m_R1, m_P1, width, height,
    m_rectification_map_left_x, m_rectification_map_left_y );
  viame::measurement::rectification_maps(
    m_K2, m_D2, m_R2, m_P2, width, height,
    m_rectification_map_right_x, m_rectification_map_right_y );

  m_rectification_valid = true;
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

  if( !m_rectification_valid )
  {
    return original_point;
  }

  auto const& K = is_right_camera ? m_K2 : m_K1;
  auto const& D = is_right_camera ? m_D2 : m_D1;
  auto const& R = is_right_camera ? m_R2 : m_R1;
  auto const& P = is_right_camera ? m_P2 : m_P1;

  return viame::measurement::undistort_point( original_point, K, D, R, P );
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

  auto const& R = is_right_camera ? m_R2 : m_R1;
  auto const& P = is_right_camera ? m_P2 : m_P1;
  auto const& K = is_right_camera ? m_K2 : m_K1;
  auto const& D = is_right_camera ? m_D2 : m_D1;

  // Extract rectified camera intrinsics from P (3x4 projection matrix)
  double fx_rect = P( 0, 0 );
  double fy_rect = P( 1, 1 );
  double cx_rect = P( 0, 2 );
  double cy_rect = P( 1, 2 );

  // Convert rectified pixel to normalized rectified coordinates
  double x_norm_rect = ( rectified_point.x() - cx_rect ) / fx_rect;
  double y_norm_rect = ( rectified_point.y() - cy_rect ) / fy_rect;

  // Apply inverse rectification rotation to get normalized original coordinates
  kv::vector_3d const pt_orig =
    R.transpose() * kv::vector_3d( x_norm_rect, y_norm_rect, 1.0 );

  double x_norm = pt_orig[ 0 ] / pt_orig[ 2 ];
  double y_norm = pt_orig[ 1 ] / pt_orig[ 2 ];

  // Apply distortion and camera matrix, which is `cv::projectPoints` with an
  // identity pose
  return viame::measurement::project_point(
    kv::vector_3d( x_norm, y_norm, 1.0 ), K, D );
}

// -----------------------------------------------------------------------------
kv::image_of< uint8_t >
map_keypoints_to_camera
::rectify_image( const kv::image_of< uint8_t >& image,
                 bool is_right_camera ) const
{
  if( !m_rectification_computed )
  {
    return image;
  }

  auto const& map_x = is_right_camera ? m_rectification_map_right_x
                                      : m_rectification_map_left_x;
  auto const& map_y = is_right_camera ? m_rectification_map_right_y
                                      : m_rectification_map_left_y;

  return io::remap( image, map_x, map_y, io::interpolation::BILINEAR,
                    io::border_mode::CONSTANT, 0.0 );
}

// -----------------------------------------------------------------------------
// Helper function to compute census transform of an image
// Census transform compares each pixel to its neighbors, creating a binary pattern
// that is robust to illumination changes
namespace {
kv::image_of< int32_t >
compute_census_transform( const kv::image_of< uint8_t >& input,
                         int window_radius = 2 )
{
  auto const gray = to_gray( input );

  kv::image_of< int32_t > census( gray.width(), gray.height(), 1 );

  for( size_t j = 0; j < census.height(); ++j )
  {
    for( size_t i = 0; i < census.width(); ++i )
    {
      census( i, j, 0 ) = 0;
    }
  }

  auto const rows = static_cast< int >( gray.height() );
  auto const cols = static_cast< int >( gray.width() );

  for( int y = window_radius; y < rows - window_radius; ++y )
  {
    for( int x = window_radius; x < cols - window_radius; ++x )
    {
      unsigned int census_val = 0;
      uint8_t center = gray( x, y, 0 );
      int bit_pos = 0;

      for( int dy = -window_radius; dy <= window_radius; ++dy )
      {
        for( int dx = -window_radius; dx <= window_radius; ++dx )
        {
          if( dx == 0 && dy == 0 ) continue;  // Skip center

          if( gray( x + dx, y + dy, 0 ) < center )
          {
            census_val |= ( 1u << bit_pos );
          }
          ++bit_pos;
        }
      }
      census( x, y, 0 ) = static_cast< int32_t >( census_val );
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
  const kv::image_of< int32_t >& census_template,
  const kv::image_of< int32_t >& census_search,
  int search_x, int search_y,
  int template_width, int template_height )
{
  int max_distance = template_width * template_height * 24;  // 24 bits max per pixel for 5x5 window
  int total_distance = 0;

  for( int ty = 0; ty < template_height; ++ty )
  {
    for( int tx = 0; tx < template_width; ++tx )
    {
      int t_val = census_template( tx, ty, 0 );
      int s_val = census_search( search_x + tx, search_y + ty, 0 );
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
  const kv::image_of< uint8_t >& source_image, int x, int y,
  prepared_template& tmpl ) const
{
  tmpl.valid = false;
  int half_template = m_template_size / 2;
  int margin = m_use_census_transform ? half_template + 2 : half_template;

  auto const cols = static_cast< int >( source_image.width() );
  auto const rows = static_cast< int >( source_image.height() );

  if( x < margin || x >= cols - margin ||
      y < margin || y >= rows - margin )
  {
    return false;
  }

  // Extract NCC template
  tmpl.ncc_template = region(
    source_image, image_rect( x - half_template, y - half_template,
                              m_template_size, m_template_size ) );

  if( m_use_census_transform )
  {
    int census_margin = 2;
    auto const template_region = region(
      source_image,
      image_rect( x - half_template - census_margin,
                  y - half_template - census_margin,
                  m_template_size + 2 * census_margin,
                  m_template_size + 2 * census_margin ) );
    auto const census_full =
      compute_census_transform( template_region, census_margin );
    tmpl.census_template = io::crop(
      census_full, static_cast< size_t >( census_margin ),
      static_cast< size_t >( census_margin ),
      static_cast< size_t >( m_template_size ),
      static_cast< size_t >( m_template_size ) );
  }

  tmpl.valid = true;
  return true;
}

// -----------------------------------------------------------------------------
double
map_keypoints_to_camera
::score_template_at_point(
  const prepared_template& tmpl,
  const kv::image_of< uint8_t >& target_image, int x, int y ) const
{
  int half_template = m_template_size / 2;
  int margin = m_use_census_transform ? half_template + 2 : half_template;

  auto const cols = static_cast< int >( target_image.width() );
  auto const rows = static_cast< int >( target_image.height() );

  if( x < margin || x >= cols - margin ||
      y < margin || y >= rows - margin )
  {
    return -1.0;
  }

  if( m_use_census_transform )
  {
    int census_margin = 2;
    auto const target_region = region(
      target_image,
      image_rect( x - half_template - census_margin,
                  y - half_template - census_margin,
                  m_template_size + 2 * census_margin,
                  m_template_size + 2 * census_margin ) );
    auto const census_target =
      compute_census_transform( target_region, census_margin );

    return census_template_match( tmpl.census_template, census_target,
                                   census_margin, census_margin,
                                   m_template_size, m_template_size );
  }
  else
  {
    auto const target_patch = region(
      target_image, image_rect( x - half_template, y - half_template,
                                m_template_size, m_template_size ) );

    auto const result = io::match_template_ncc( target_patch,
                                                tmpl.ncc_template );
    return static_cast< double >( result( 0, 0, 0 ) );
  }
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_template_matching(
  const kv::image_of< uint8_t >& left_image_rect,
  const kv::image_of< uint8_t >& right_image_rect,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect,
  const kv::image_container_sptr& disparity_map ) const
{
  auto const right_cols = static_cast< int >( right_image_rect.width() );
  auto const right_rows = static_cast< int >( right_image_rect.height() );

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
  else if( m_use_disparity_hint && disparity_map )
  {
    // Sample SGBM disparity map near the query point
    // Average over a small window for robustness
    auto const& hint = disparity_map->get_image();
    kv::image_of< int16_t > const disparity( hint );

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

        if( sample_x >= 0 &&
            sample_x < static_cast< int >( disparity.width() ) &&
            sample_y >= 0 &&
            sample_y < static_cast< int >( disparity.height() ) )
        {
          int16_t disp_raw = disparity( sample_x, sample_y, 0 );
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
    else if( m_rectification_valid && m_default_depth > 0 )
    {
      // Fall back to default depth computation
      expected_disparity = -m_P2( 0, 3 ) / m_default_depth;
    }
  }
  else if( m_rectification_valid && m_default_depth > 0 )
  {
    // Compute disparity from default depth using camera parameters
    expected_disparity = -m_P2( 0, 3 ) / m_default_depth;
  }

  // Compute expected right x position based on disparity
  int expected_right_x = static_cast< int >( x_left - expected_disparity );

  // Define search region centered around expected position
  // Use half the search range on each side of expected position for efficiency
  int half_search = m_search_range / 2;
  int search_min_x = std::max( margin, expected_right_x - half_search );
  int search_max_x = std::min( right_cols - margin, expected_right_x + half_search );

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
  search_max_y = std::min( right_rows - margin, search_max_y );

  if( search_max_y < search_min_y )
  {
    return false;
  }

  double max_val = -1.0;
  int max_loc_x = 0;
  int max_loc_y = 0;

  if( m_use_census_transform )
  {
    // Census transform based matching (uses prepared census template)
    int census_margin = 2;

    // Compute census transform of search region
    image_rect search_rect_ext(
      search_min_x - half_template - census_margin,
      search_min_y - half_template - census_margin,
      ( search_max_x - search_min_x ) + m_template_size + 2 * census_margin,
      ( search_max_y - search_min_y ) + m_template_size + 2 * census_margin );

    // Bounds check
    if( search_rect_ext.x < 0 || search_rect_ext.y < 0 ||
        search_rect_ext.x + search_rect_ext.width > right_cols ||
        search_rect_ext.y + search_rect_ext.height > right_rows )
    {
      return false;
    }

    auto const search_region = region( right_image_rect, search_rect_ext );
    auto const census_search =
      compute_census_transform( search_region, census_margin );

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
          max_loc_x = sx;
          max_loc_y = sy;
        }
      }
    }

    // Convert the best location to image coordinates
    right_point_rect = kv::vector_2d(
      search_min_x + max_loc_x,
      search_min_y + max_loc_y );
  }
  else
  {
    // Standard intensity-based template matching (uses prepared NCC template)

    // Define search region including epipolar band
    int search_height = ( search_max_y - search_min_y ) + m_template_size;
    image_rect search_rect( search_min_x - half_template,
                            search_min_y - half_template,
                            search_max_x - search_min_x + m_template_size,
                            search_height );

    // Check search rect validity
    if( search_rect.x < 0 || search_rect.y < 0 ||
        search_rect.x + search_rect.width > right_cols ||
        search_rect.y + search_rect.height > right_rows )
    {
      return false;
    }

    auto const search_region = region( right_image_rect, search_rect );
    auto const result = io::match_template_ncc( search_region,
                                                tmpl.ncc_template );

    auto const result_cols = static_cast< int >( result.width() );
    auto const result_rows = static_cast< int >( result.height() );

    if( m_use_multires_search &&
        search_rect.width > m_template_size + m_multires_coarse_step * 4 )
    {
      // Multi-resolution search: coarse pass then fine pass
      double coarse_max_val = -1.0;
      int coarse_max_x = 0;
      int coarse_max_y = 0;

      for( int ry = 0; ry < result_rows; ++ry )
      {
        for( int rx = 0; rx < result_cols; rx += m_multires_coarse_step )
        {
          double val = result( rx, ry, 0 );
          if( val > coarse_max_val )
          {
            coarse_max_val = val;
            coarse_max_x = rx;
            coarse_max_y = ry;
          }
        }
      }

      // Fine pass: search around the coarse best match
      int fine_half_range = m_multires_coarse_step * 2;
      int fine_min_x = std::max( 0, coarse_max_x - fine_half_range );
      int fine_max_x = std::min( result_cols - 1, coarse_max_x + fine_half_range );
      int fine_min_y = std::max( 0, coarse_max_y - fine_half_range );
      int fine_max_y = std::min( result_rows - 1, coarse_max_y + fine_half_range );

      max_val = coarse_max_val;
      max_loc_x = coarse_max_x;
      max_loc_y = coarse_max_y;

      for( int ry = fine_min_y; ry <= fine_max_y; ++ry )
      {
        for( int rx = fine_min_x; rx <= fine_max_x; ++rx )
        {
          double val = result( rx, ry, 0 );
          if( val > max_val )
          {
            max_val = val;
            max_loc_x = rx;
            max_loc_y = ry;
          }
        }
      }
    }
    else
    {
      // Standard single-pass template matching, and `cv::minMaxLoc` over it
      for( int ry = 0; ry < result_rows; ++ry )
      {
        for( int rx = 0; rx < result_cols; ++rx )
        {
          double val = result( rx, ry, 0 );
          if( val > max_val )
          {
            max_val = val;
            max_loc_x = rx;
            max_loc_y = ry;
          }
        }
      }
    }

    // Convert the best location to image coordinates
    right_point_rect = kv::vector_2d(
      search_rect.x + max_loc_x + half_template,
      search_rect.y + max_loc_y + half_template );
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
  const kv::image_of< uint8_t >& source_image,
  const kv::image_of< uint8_t >& target_image,
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
  double second_best_score = -1.0;
  kv::vector_2d best_point;

  // Minimum pixel distance between best and second-best to be considered
  // distinct candidates (avoids penalizing neighboring epipolar samples
  // that are essentially the same match)
  const double min_distinct_dist_sq = m_template_size * m_template_size;

  for( const auto& ep_pt : epipolar_points )
  {
    int x_tgt = static_cast< int >( ep_pt.x() + 0.5 );
    int y_tgt = static_cast< int >( ep_pt.y() + 0.5 );

    double score = score_template_at_point( tmpl, target_image, x_tgt, y_tgt );
    if( score > best_score )
    {
      // Check if previous best is far enough to count as second-best
      if( best_score > 0 )
      {
        double dx = ep_pt.x() - best_point.x();
        double dy = ep_pt.y() - best_point.y();
        if( dx * dx + dy * dy >= min_distinct_dist_sq )
        {
          second_best_score = best_score;
        }
      }
      best_score = score;
      best_point = ep_pt;
    }
    else if( score > second_best_score )
    {
      double dx = ep_pt.x() - best_point.x();
      double dy = ep_pt.y() - best_point.y();
      if( dx * dx + dy * dy >= min_distinct_dist_sq )
      {
        second_best_score = score;
      }
    }
  }

  if( best_score < m_template_matching_threshold )
  {
    LOG_INFO( logger, "NCC REJECT: threshold best_score=" << best_score
      << " < " << m_template_matching_threshold
      << " second=" << second_best_score
      << " src=(" << source_point.x() << "," << source_point.y() << ")"
      << " n_pts=" << epipolar_points.size() );
    return false;
  }

  // Uniqueness ratio test: reject if second-best is too close to best
  if( m_uniqueness_ratio > 0 && second_best_score > 0 && best_score > 0 )
  {
    double ratio = second_best_score / best_score;
    if( ratio > m_uniqueness_ratio )
    {
      LOG_INFO( logger, "NCC REJECT: uniqueness best=" << best_score
        << " second=" << second_best_score
        << " ratio=" << ratio << " > " << m_uniqueness_ratio
        << " src=(" << source_point.x() << "," << source_point.y() << ")"
        << " best_pt=(" << best_point.x() << "," << best_point.y() << ")" );
      return false;
    }
  }

  LOG_INFO( logger, "NCC ACCEPT: best=" << best_score
    << " second=" << second_best_score
    << " src=(" << source_point.x() << "," << source_point.y() << ")"
    << " match=(" << best_point.x() << "," << best_point.y() << ")" );

  target_point = best_point;
  return true;
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_epipolar_strip_ncc(
  const kv::image_of< uint8_t >& source_image,
  const kv::image_of< uint8_t >& target_image,
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
  strip_x2 = std::min( static_cast< int >( target_image.width() ) - 1, strip_x2 );
  strip_y2 = std::min( static_cast< int >( target_image.height() ) - 1, strip_y2 );

  int strip_w = strip_x2 - strip_x + 1;
  int strip_h = strip_y2 - strip_y + 1;

  // Strip must be at least as large as the template
  if( strip_w < m_template_size || strip_h < m_template_size )
  {
    return false;
  }

  // Extract strip subimage and correlate the template over it
  auto const strip = region( target_image,
                             image_rect( strip_x, strip_y, strip_w, strip_h ) );

  auto const result = io::match_template_ncc( strip, tmpl.ncc_template );

  auto const result_cols = static_cast< int >( result.width() );
  auto const result_rows = static_cast< int >( result.height() );

  // `cv::minMaxLoc` over the correlation surface
  double max_val = -std::numeric_limits< double >::max();
  int max_loc_x = 0;
  int max_loc_y = 0;

  for( int ry = 0; ry < result_rows; ++ry )
  {
    for( int rx = 0; rx < result_cols; ++rx )
    {
      double const val = result( rx, ry, 0 );

      if( val > max_val )
      {
        max_val = val;
        max_loc_x = rx;
        max_loc_y = ry;
      }
    }
  }

  if( max_val < m_template_matching_threshold )
  {
    return false;
  }

  // Uniqueness ratio test: find second-best peak at least template_size away
  if( m_uniqueness_ratio > 0 )
  {
    // Suppress the neighborhood around the best match
    int suppress_radius = m_template_size;
    int sr_x1 = std::max( 0, max_loc_x - suppress_radius );
    int sr_y1 = std::max( 0, max_loc_y - suppress_radius );
    int sr_x2 = std::min( result_cols - 1, max_loc_x + suppress_radius );
    int sr_y2 = std::min( result_rows - 1, max_loc_y + suppress_radius );

    double second_max_val = -std::numeric_limits< double >::max();

    for( int ry = 0; ry < result_rows; ++ry )
    {
      for( int rx = 0; rx < result_cols; ++rx )
      {
        if( rx >= sr_x1 && rx <= sr_x2 && ry >= sr_y1 && ry <= sr_y2 )
        {
          continue;
        }

        second_max_val = std::max< double >( second_max_val,
                                             result( rx, ry, 0 ) );
      }
    }

    if( second_max_val > 0 && max_val > 0 )
    {
      double ratio = second_max_val / max_val;
      if( ratio > m_uniqueness_ratio )
      {
        return false;
      }
    }
  }

  // Convert result location to image coordinates
  // the correlation surface's offset is the top left of the template placement
  double match_x = strip_x + max_loc_x + half_template;
  double match_y = strip_y + max_loc_y + half_template;

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
kv::image_container_sptr
map_keypoints_to_camera
::compute_sgbm_disparity(
  const kv::image_of< uint8_t >& left_image_rect,
  const kv::image_of< uint8_t >& right_image_rect )
{
  if( !m_stereo_depth_map_algorithm )
  {
    // Algorithm not configured, return nothing
    return nullptr;
  }

  kv::image_container_sptr left_container =
    std::make_shared< kv::simple_image_container >( left_image_rect );
  kv::image_container_sptr right_container =
    std::make_shared< kv::simple_image_container >( right_image_rect );

  return m_stereo_depth_map_algorithm->compute( left_container,
                                                right_container );
}

// -----------------------------------------------------------------------------
bool
map_keypoints_to_camera
::find_corresponding_point_sgbm(
  const kv::image_container_sptr& disparity_map,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect ) const
{
  if( !disparity_map )
  {
    return false;
  }

  kv::image_of< int16_t > const disparity_image( disparity_map->get_image() );

  int x = static_cast< int >( left_point_rect.x() + 0.5 );
  int y = static_cast< int >( left_point_rect.y() + 0.5 );

  // Check bounds
  if( x < 0 || x >= static_cast< int >( disparity_image.width() ) ||
      y < 0 || y >= static_cast< int >( disparity_image.height() ) )
  {
    return false;
  }

  // Get disparity value (SGBM returns fixed-point values scaled by 16)
  int16_t disp_raw = disparity_image( x, y, 0 );

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
const kv::image_of< float >&
map_keypoints_to_camera
::get_rectification_map_x( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_x : m_rectification_map_left_x;
}

// -----------------------------------------------------------------------------
const kv::image_of< float >&
map_keypoints_to_camera
::get_rectification_map_y( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_y : m_rectification_map_left_y;
}

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

#endif // !VIAME_MEASUREMENT_PYTHON_BINDINGS

// =============================================================================
// Python bindings
//
// Compiled only into the viame.core._measurement Python module (the module
// target is built with VIAME_MEASUREMENT_PYTHON_BINDINGS defined), never into
// the viame_core library even though both targets compile this file. The module
// links viame_core for the actual implementations and only wraps them here, so
// the stereo measurement / aggregation math is not duplicated.
// =============================================================================
#ifdef VIAME_MEASUREMENT_PYTHON_BINDINGS

#include "camera_rig_io.h"

#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/rotation.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
// Inputs are passed as flat std::vector<double> rather than Eigen/numpy types:
// pybind11's Eigen<->numpy caster is incompatible with numpy 2.0 (it crashes in
// the numpy C-API), whereas the STL caster uses only the CPython sequence API.
kv::matrix_3x3d
to_matrix_3x3( std::vector< double > const& v, char const* name )
{
  if( v.size() != 9 )
  {
    throw std::invalid_argument(
      std::string( name ) + " must have 9 elements (row-major 3x3)" );
  }
  kv::matrix_3x3d m;
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      m( i, j ) = v[ i * 3 + j ];
    }
  }
  return m;
}

kv::vector_3d
to_vector_3d( std::vector< double > const& v, char const* name )
{
  if( v.size() != 3 )
  {
    throw std::invalid_argument(
      std::string( name ) + " must have 3 elements" );
  }
  return kv::vector_3d( v[ 0 ], v[ 1 ], v[ 2 ] );
}

kv::vector_2d
to_vector_2d( std::vector< double > const& v, char const* name )
{
  if( v.size() != 2 )
  {
    throw std::invalid_argument(
      std::string( name ) + " must have 2 elements" );
  }
  return kv::vector_2d( v[ 0 ], v[ 1 ] );
}

// ----------------------------------------------------------------------------
/// Build left/right cameras from a stereo calibration and run the C++ stereo
/// measurement on a line's two endpoints.
///
/// The left camera is the calibration origin (identity pose); the right camera
/// is positioned by the calibration extrinsics, where a left-frame point X maps
/// to the right frame as X_right = rotation * X + translation. KWIVER's camera
/// projects as x = K * R * (X - center), so center = -R^T * translation.
py::dict
compute_stereo_measurement_from_calibration(
  std::vector< double > const& k_left,
  std::vector< double > const& k_right,
  std::vector< double > const& rotation,
  std::vector< double > const& translation,
  std::vector< double > const& left_head,
  std::vector< double > const& right_head,
  std::vector< double > const& left_tail,
  std::vector< double > const& right_tail )
{
  kv::matrix_3x3d const mat_k_left = to_matrix_3x3( k_left, "k_left" );
  kv::matrix_3x3d const mat_k_right = to_matrix_3x3( k_right, "k_right" );
  kv::matrix_3x3d const mat_rotation = to_matrix_3x3( rotation, "rotation" );
  kv::vector_3d const vec_translation = to_vector_3d( translation, "translation" );

  auto const intrinsics_left =
    std::make_shared< kv::simple_camera_intrinsics >( mat_k_left );
  auto const intrinsics_right =
    std::make_shared< kv::simple_camera_intrinsics >( mat_k_right );

  // Concrete matrix/vector temporaries to avoid ambiguous Eigen-expression
  // overloads of the rotation_d / vector_3d constructors.
  kv::matrix_3x3d const identity_rotation = kv::matrix_3x3d::Identity();
  kv::vector_3d const center_right = -mat_rotation.transpose() * vec_translation;

  kv::simple_camera_perspective const left_cam(
    kv::vector_3d( 0.0, 0.0, 0.0 ),
    kv::rotation_d( identity_rotation ),
    intrinsics_left );

  kv::simple_camera_perspective const right_cam(
    center_right,
    kv::rotation_d( mat_rotation ),
    intrinsics_right );

  auto const m = viame::core::compute_stereo_measurement(
    left_cam, right_cam,
    to_vector_2d( left_head, "left_head" ),
    to_vector_2d( right_head, "right_head" ),
    to_vector_2d( left_tail, "left_tail" ),
    to_vector_2d( right_tail, "right_tail" ) );

  py::dict result;
  result[ "length" ] = m.length;
  result[ "midpoint_x" ] = m.x;
  result[ "midpoint_y" ] = m.y;
  result[ "midpoint_z" ] = m.z;
  result[ "midpoint_range" ] = m.range;
  result[ "stereo_rms" ] = m.rms;
  return result;
}

// ----------------------------------------------------------------------------
// Flatten a 3x3 matrix to a row-major std::vector<double> of length 9.
std::vector< double >
flatten_3x3( kv::matrix_3x3d const& mat )
{
  std::vector< double > out( 9 );
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      out[ i * 3 + j ] = mat( i, j );
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
/// Load a stereo calibration file using viame::core::read_stereo_rig (the same
/// loader the measurement pipeline processes use) and return the intrinsics and
/// the right-relative-to-left extrinsics as flat lists. Supports every format
/// read_stereo_rig handles: .json, .yml/.yaml, .npz and OpenCV directories.
py::dict
load_stereo_calibration( std::string const& path )
{
  auto const rig = viame::read_stereo_rig( path );
  if( !rig )
  {
    throw std::runtime_error( "Could not read stereo calibration from: " + path );
  }

  auto const left =
    std::dynamic_pointer_cast< kv::camera_perspective >( rig->left() );
  auto const right =
    std::dynamic_pointer_cast< kv::camera_perspective >( rig->right() );
  if( !left || !right )
  {
    throw std::runtime_error(
      "Stereo calibration does not contain perspective cameras: " + path );
  }

  kv::matrix_3x3d const r_left = left->rotation().matrix();
  kv::matrix_3x3d const r_right = right->rotation().matrix();
  kv::vector_3d const c_left = left->center();
  kv::vector_3d const c_right = right->center();

  // Right camera relative to the left (which is the measurement reference):
  // X_right = R * X_left + T. Computed from absolute poses so it is correct
  // even if the left camera is not at the identity pose.
  kv::matrix_3x3d const rotation = r_right * r_left.transpose();
  kv::vector_3d const translation = r_right * ( c_left - c_right );

  // Distortion coefficients (radial-tangential [k1,k2,p1,p2,k3,...]), exactly
  // as read_stereo_rig parsed them into the camera intrinsics.
  auto const dist_to_vec =
    []( kv::camera_intrinsics_sptr const& ci ) -> std::vector< double >
    {
      std::vector< double > out;
      if( ci )
      {
        auto const& d = ci->dist_coeffs();
        out.reserve( d.size() );
        for( size_t i = 0; i < d.size(); ++i )
        {
          out.push_back( d[ i ] );
        }
      }
      return out;
    };

  py::dict result;
  result[ "k_left" ] = flatten_3x3( left->intrinsics()->as_matrix() );
  result[ "k_right" ] = flatten_3x3( right->intrinsics()->as_matrix() );
  result[ "dist_left" ] = dist_to_vec( left->intrinsics() );
  result[ "dist_right" ] = dist_to_vec( right->intrinsics() );
  result[ "rotation" ] = flatten_3x3( rotation );
  result[ "translation" ] = std::vector< double >{
    translation[ 0 ], translation[ 1 ], translation[ 2 ] };
  return result;
}

} // namespace <anonymous>

// ----------------------------------------------------------------------------
PYBIND11_MODULE( _measurement, m )
{
  m.doc() =
    "VIAME stereo measurement bindings "
    "(wraps viame::core::compute_stereo_measurement).";

  m.def(
    "compute_stereo_measurement_from_calibration",
    &compute_stereo_measurement_from_calibration,
    py::arg( "k_left" ), py::arg( "k_right" ),
    py::arg( "rotation" ), py::arg( "translation" ),
    py::arg( "left_head" ), py::arg( "right_head" ),
    py::arg( "left_tail" ), py::arg( "right_tail" ),
    "Compute the full stereo measurement (length, 3D midpoint, range, RMS) "
    "for a line's two endpoints given the stereo calibration. Returns a dict "
    "with keys: length, midpoint_x, midpoint_y, midpoint_z, midpoint_range, "
    "stereo_rms. Values are in calibration units." );

  m.def(
    "aggregate_lengths",
    &viame::core::aggregate_lengths,
    py::arg( "lengths" ),
    py::arg( "method" ) = "average",
    py::arg( "iqr_factor" ) = 1.5,
    "Aggregate per-detection lengths along a track into a single value. "
    "method: 'average' (mean, default), 'average_iqr' (IQR-trimmed mean) or "
    "'median'. Returns the aggregated length, or -1 if there are none. Shares "
    "the implementation used by the pair_stereo_tracks pipeline process." );

  m.def(
    "load_stereo_calibration",
    &load_stereo_calibration,
    py::arg( "path" ),
    "Load a stereo calibration file via viame::core::read_stereo_rig (the same "
    "loader the measurement pipeline processes use; supports .json, .yml/.yaml, "
    ".mat, .npz and OpenCV calibration directories). Returns a dict with flat "
    "row-major k_left, k_right, the radial-tangential dist_left/dist_right "
    "coefficients ([k1,k2,p1,p2,k3,...]), rotation (right relative to left) "
    "and translation." );
}

#endif // VIAME_MEASUREMENT_PYTHON_BINDINGS
