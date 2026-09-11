#include "pair_stereo_detections.h"

#include "../core/camera_rig_io.h"

#include <viame/core_types/camera_intrinsics.h>
#include <viame/core_types/camera_perspective.h>

#include <cmath>
#include <limits>

namespace
{

namespace mp = viame::measurement;
namespace kv = kwiver::vital;

/// The intersection of two rectangles, which is `cv::Rect`'s `operator&`:
/// an empty rectangle when they do not meet.
viame::image_rect
intersect( viame::image_rect const& a, viame::image_rect const& b )
{
  int const x = std::max( a.x, b.x );
  int const y = std::max( a.y, b.y );
  int const right = std::min( a.x + a.width, b.x + b.width );
  int const bottom = std::min( a.y + a.height, b.y + b.height );

  if( right <= x || bottom <= y )
  {
    return viame::image_rect( 0, 0, 0, 0 );
  }

  return viame::image_rect( x, y, right - x, bottom - y );
}

/// The camera intrinsics of one side of a rig, as a matrix and coefficients.
void
intrinsics_of( kv::camera_perspective const& camera,
               kv::matrix_3x3d& matrix, mp::distortion_t& coefficients )
{
  auto const intrinsics = camera.intrinsics();

  matrix = intrinsics->as_matrix();
  coefficients = intrinsics->dist_coeffs();

  // `cv::Mat::zeros( 5, 1 )` is what the OpenCV path produced for a rig with
  // none, and an empty vector is what `projection` calls the same thing.
  if( coefficients.size() < 4 )
  {
    coefficients.clear();
  }
}

} // namespace

void
viame::pair_stereo_detections
::load_camera_calibration()
{
  // `plugins/core/camera_rig_io` rather than
  // `calibrate_stereo_cameras::load_calibration`, which was the same four
  // readers with OpenCV types in the middle. P7-T05 already made the YAML
  // reader in-house and P7-T06 deleted the OpenCV one, so this is the only
  // remaining copy.
  auto const rig = viame::read_stereo_rig( m_calibration_file );

  if( !rig || !rig->left() || !rig->right() )
  {
    VITAL_THROW( kwiver::vital::invalid_data,
                 "Could not read calibration : " + m_calibration_file );
  }

  auto const left =
    std::dynamic_pointer_cast< kv::camera_perspective >( rig->left() );
  auto const right =
    std::dynamic_pointer_cast< kv::camera_perspective >( rig->right() );

  if( !left || !right )
  {
    VITAL_THROW( kwiver::vital::invalid_data,
                 "Calibration is not a pair of perspective cameras : " +
                   m_calibration_file );
  }

  intrinsics_of( *left, m_K1, m_D1 );
  intrinsics_of( *right, m_K2, m_D2 );

  // The rig stores each camera's pose in a common frame; what the pairing
  // wants is right relative to left.
  auto const rotation_left = left->rotation().matrix();
  auto const rotation_right = right->rotation().matrix();

  m_R = rotation_right * rotation_left.transpose();
  m_T = rotation_right * ( left->center() - right->center() );

  // Compute the rotation vector from the matrix for later use
  m_Rvec = mp::inverse_rodrigues( m_R );

  m_rectified = false;
}

float
viame::pair_stereo_detections
::compute_median( std::vector< float > values, bool is_sorted )
{
  float median = 0;
  size_t size = values.size();
  if( size > 0 )
  {
    if( !is_sorted )
    {
      std::sort( values.begin(), values.end() );
    }

    if( size % 2 == 0 )
    {
      median = ( values[size / 2 - 1] + values[size / 2] ) / 2;
    }
    else
    {
      median = values[size / 2];
    }
  }
  return median;
}

viame::image_rect
viame::pair_stereo_detections
::bbox_to_mask_rect( const kwiver::vital::bounding_box_d& bbox )
{
  // `cv::Rect` from two points truncates each toward zero and takes the
  // difference, which is what this does.
  int const x = static_cast< int >( bbox.upper_left().x() );
  int const y = static_cast< int >( bbox.upper_left().y() );

  return image_rect( x, y,
                     static_cast< int >( bbox.lower_right().x() ) - x,
                     static_cast< int >( bbox.lower_right().y() ) - y );
}


kwiver::vital::bounding_box_d
viame::pair_stereo_detections
::mask_rect_to_bbox( const image_rect& rect )
{
  return { kwiver::vital::vector_2d( rect.x, rect.y ),
           kwiver::vital::vector_2d( rect.x + rect.width,
                                     rect.y + rect.height ) };
}


kwiver::vital::image_of< uint8_t >
viame::pair_stereo_detections
::get_standard_mask( const kwiver::vital::detected_object_sptr& det )
{
  auto vital_mask = det->mask();
  if( !vital_mask )
  {
    return {};
  }

  kv::image_of< uint8_t > const mask( vital_mask->get_image() );

  auto const box = bbox_to_mask_rect( det->bounding_box() );
  auto const width = static_cast< size_t >( std::max( 0, box.width ) );
  auto const height = static_cast< size_t >( std::max( 0, box.height ) );

  if( mask.width() == width && mask.height() == height )
  {
    return mask;
  }

  // A mask that does not fill its box is placed in the top left of one that
  // does, zero elsewhere.
  kv::image_of< uint8_t > standard_mask( width, height, 1 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      standard_mask( i, j, 0 ) =
        ( i < mask.width() && j < mask.height() ) ? mask( i, j, 0 ) : 0;
    }
  }

  return standard_mask;
}

inline void print( const kwiver::vital::bounding_box_d& bbox, const std::string& context = "" )
{
  if( !bbox.is_valid() )
  {
    std::cout << context << " - kv::bounding_box_d( INVALID )" << std::endl;
    return;
  }

  std::cout << context << " - kv::bounding_box_d( upperLeft {"
            << bbox.upper_left().x() << ", " << bbox.upper_left().y()
            << "}, lowerRight {" << bbox.lower_right().x() << ", "
            << bbox.lower_right().y() << "})" << std::endl;
}

inline void print( const kwiver::vital::vector_< 2, double >& mat, const std::string& context )
{
  std::cout << context << " {" << mat.x() << ", " << mat.y() << "}" << std::endl;
}


inline void print( const kwiver::vital::vector_< 3, double >& mat, const std::string& context )
{
  std::cout << context << " {" << mat[0] << ", " << mat[1] << ", " << mat[2] << "}" << std::endl;
}

inline void print( const viame::image_rect& bbox, const std::string& context = "" )
{
  std::cout << context << " - image_rect( upperLeft {" << bbox.x << ", "
            << bbox.y << "}, lowerRight {" << bbox.x + bbox.width << ", "
            << bbox.y + bbox.height << "})" << std::endl;
}


viame::Detections3DPositions
viame::pair_stereo_detections
::estimate_3d_position_from_detection(
    const kwiver::vital::detected_object_sptr& detection,
    const kwiver::vital::image_of< float >& pos_3d_map,
    bool do_undistort_points,
    float bbox_crop_ratio ) const
{
  // Extract mask and corresponding mask bounding box from input detection
  auto mask = get_standard_mask( detection );

  // If mask is invalid return the estimated position from bounding box center
  if( mask.size() == 0 )
  {
    return estimate_3d_position_from_bbox( detection->bounding_box(), pos_3d_map,
                                           bbox_crop_ratio, do_undistort_points );
  }

  // Otherwise, returns average 3D distance for each point in mask
  return estimate_3d_position_from_unrectified_mask( detection->bounding_box(), pos_3d_map,
                                                     mask, do_undistort_points );
}

kwiver::vital::bounding_box_d
viame::pair_stereo_detections
::get_rectified_bbox( const kwiver::vital::bounding_box_d& bbox,
                      bool is_left_image ) const
{
  const auto tl = undistort_point( kv::vector_2d( bbox.upper_left().x(), bbox.upper_left().y() ), is_left_image );
  const auto br = undistort_point( kv::vector_2d( bbox.lower_right().x(), bbox.lower_right().y() ), is_left_image );
  return { tl.x(), tl.y(), br.x(), br.y() };
}

bool
viame::pair_stereo_detections
::point_is_valid( float x, float y, float z )
{
  return ( ( z > 0 ) && std::isfinite( x ) && std::isfinite( y ) && std::isfinite( z ) );
}

bool
viame::pair_stereo_detections
::point_is_valid( const kwiver::vital::vector_3d& pt )
{
  return point_is_valid( static_cast< float >( pt[ 0 ] ),
                         static_cast< float >( pt[ 1 ] ),
                         static_cast< float >( pt[ 2 ] ) );
}

viame::Detections3DPositions
viame::pair_stereo_detections
::estimate_3d_position_from_bbox( const kwiver::vital::bounding_box_d& bbox,
                                  const kwiver::vital::image_of< float >& pos_3d_map,
                                  float crop_ratio,
                                  bool do_undistort_points ) const
{
  const auto rectified_bbox = do_undistort_points ? get_rectified_bbox( bbox, true ) : bbox;

  // depth from median of values in the center part of the bounding box
  float crop_width = crop_ratio * ( float ) rectified_bbox.width();
  float crop_height = crop_ratio * ( float ) rectified_bbox.height();
  image_rect crop_rect{ ( int ) ( rectified_bbox.center().x() - crop_width / 2 ),
                        ( int ) ( rectified_bbox.center().y() - crop_height / 2 ),
                        ( int ) crop_width, ( int ) crop_height };

  // Intersect crop rectangle with 3D map rect to avoid out of bounds crop
  crop_rect = intersect( crop_rect,
                         image_rect( 0, 0,
                                     ( int ) pos_3d_map.width(),
                                     ( int ) pos_3d_map.height() ) );

  if( m_verbose )
  {
    print( crop_rect, "CROP RECT" );
  }

  // If resized crop is out of the 3D map (detection out of left / right ROI overlap) return 0
  if( crop_rect.width == 0 || crop_rect.height == 0 )
  {
    return {};
  }

  // Select for valid points (with z > 0 and z != inf ) and compute xs, ys, zs
  // median from those. `cv::split` and two `reshape`s did this by walking the
  // crop in row order, which is what the loop below does.
  std::vector< float > valid_xs, valid_ys, valid_zs;

  for( int j = 0; j < crop_rect.height; ++j )
  {
    for( int i = 0; i < crop_rect.width; ++i )
    {
      auto const px = static_cast< size_t >( crop_rect.x + i );
      auto const py = static_cast< size_t >( crop_rect.y + j );

      auto const x = pos_3d_map( px, py, 0 );
      auto const y = pos_3d_map( px, py, 1 );
      auto const z = pos_3d_map( px, py, 2 );

      if( point_is_valid( x, y, z ) )
      {
        valid_xs.push_back( x );
        valid_ys.push_back( y );
        valid_zs.push_back( z );
      }
    }
  }

  // Return 3d position
  auto score = ( float ) valid_xs.size() / ( crop_width * crop_height );
  return create_3d_position( valid_xs, valid_ys, valid_zs, rectified_bbox, pos_3d_map, score );
}

viame::Detections3DPositions
viame::pair_stereo_detections
::estimate_3d_position_from_unrectified_mask(
    const kwiver::vital::bounding_box_d& bbox,
    const kwiver::vital::image_of< float >& pos_3d_map,
    const kwiver::vital::image_of< uint8_t >& mask,
    bool do_undistort_points ) const
{
  // Early return if bbox crop is out of the 3D map
  if( bbox.width() == 0 || bbox.height() == 0 )
  {
    return {};
  }

  // Find all distorted positions where mask is not empty
  std::vector< kv::vector_2d > mask_distorted_coords;
  const auto mask_tl = bbox.upper_left();
  for( size_t i_x = 0; i_x < mask.width(); i_x++ )
  {
    for( size_t i_y = 0; i_y < mask.height(); i_y++ )
    {
      if( mask( i_x, i_y, 0 ) > 0 )
      {
        mask_distorted_coords.emplace_back(
          kv::vector_2d( mask_tl.x() + i_x, mask_tl.y() + i_y ) );
      }
    }
  }

  // If no segmentation, early return
  if( mask_distorted_coords.empty() )
  {
    return {};
  }

  // Undistort mask points
  auto undistorted_mask_coords = do_undistort_points ? undistort_point( mask_distorted_coords, true )
                                                     : mask_distorted_coords;

  const auto rectified_bbox = do_undistort_points ? get_rectified_bbox( bbox, true ) : bbox;
  return estimate_3d_position_from_point_coordinates( rectified_bbox, undistorted_mask_coords, pos_3d_map );
}

viame::Detections3DPositions
viame::pair_stereo_detections
::estimate_3d_position_from_point_coordinates(
    const kwiver::vital::bounding_box_d& rectified_bbox,
    const std::vector< kwiver::vital::vector_2d >& undistorted_mask_coords,
    const kwiver::vital::image_of< float >& pos_3d_map ) const
{
  // Early return if no segmentation points
  if( undistorted_mask_coords.empty() )
  {
    return {};
  }

  // For each undistorted point coordinates, find 3D position corresponding to undistorted point
  const auto is_out_of_bounds = [&pos_3d_map]( const kv::vector_2d& pt )
  {
    return ( pt.x() < 0. ) || ( pt.y() < 0. ) ||
           ( pt.x() >= ( double ) pos_3d_map.width() ) ||
           ( pt.y() >= ( double ) pos_3d_map.height() );
  };

  int n_total{};
  std::vector< float > xs, ys, zs;
  for( const auto& point : undistorted_mask_coords )
  {
    if( is_out_of_bounds( point ) )
    {
      continue;
    }

    n_total += 1;

    auto const px = static_cast< size_t >( point.x() );
    auto const py = static_cast< size_t >( point.y() );

    auto const x = pos_3d_map( px, py, 0 );
    auto const y = pos_3d_map( px, py, 1 );
    auto const z = pos_3d_map( px, py, 2 );

    if( point_is_valid( x, y, z ) )
    {
      xs.push_back( x );
      ys.push_back( y );
      zs.push_back( z );
    }
  }

  // Return score based on number of valid position pixels vs number of mask pixels
  auto score = n_total > 0 ? ( ( float ) xs.size() / ( float ) n_total ) : 0.f;
  return create_3d_position( xs, ys, zs, rectified_bbox, pos_3d_map, score );
}

viame::Detections3DPositions
viame::pair_stereo_detections
::create_3d_position( const std::vector< float >& xs,
                      const std::vector< float >& ys,
                      const std::vector< float >& zs,
                      const kwiver::vital::bounding_box_d& bbox,
                      const kwiver::vital::image_of< float >& pos_3d_map,
                      float score ) const
{
  const auto saturate_corner = [&pos_3d_map]( const kwiver::vital::vector_< 2, double >& corner )
  {
    return kwiver::vital::vector_< 2, double >{
      std::max( 0., std::min( pos_3d_map.width() - 1., corner.x() ) ),
      std::max( 0., std::min( pos_3d_map.height() - 1., corner.y() ) ) };
  };

  const auto saturate_bbox = [&saturate_corner]( const kwiver::vital::bounding_box_d& bbox )
  {
    if( !bbox.is_valid() )
    {
      return bbox;
    }

    return kwiver::vital::bounding_box_d{ saturate_corner( bbox.upper_left() ),
                                          saturate_corner( bbox.lower_right() ) };
  };

  const auto extract_3d_bbox_from_point_list = [&]
  {
    // Find bounding box of input
    kv::vector_3d tl_3d{ std::numeric_limits< float >::max(),
                         std::numeric_limits< float >::max(),
                         std::numeric_limits< float >::max() };
    kv::vector_3d br_3d{ std::numeric_limits< float >::lowest(),
                         std::numeric_limits< float >::lowest(),
                         std::numeric_limits< float >::lowest() };

    for( size_t i_pt = 0; i_pt < xs.size(); i_pt++ )
    {
      tl_3d[0] = std::min( tl_3d[0], ( double ) xs[i_pt] );
      tl_3d[1] = std::min( tl_3d[1], ( double ) ys[i_pt] );
      tl_3d[2] = std::min( tl_3d[2], ( double ) zs[i_pt] );

      br_3d[0] = std::max( br_3d[0], ( double ) xs[i_pt] );
      br_3d[1] = std::max( br_3d[1], ( double ) ys[i_pt] );
      br_3d[2] = std::max( br_3d[2], ( double ) zs[i_pt] );
    }
    return std::vector< kv::vector_3d >{ tl_3d, br_3d };
  };

  Detections3DPositions position;
  position.score = score;
  position.center3d = kv::vector_3d{ compute_median( xs ), compute_median( ys ),
                                     compute_median( zs ) };
  position.rectified_left_bbox = saturate_bbox( bbox );
  position.left_bbox_proj_to_right_image = saturate_bbox(
    xs.empty() ?
    project_to_right_image( bbox, pos_3d_map ) :
    project_to_right_image( extract_3d_bbox_from_point_list() ) );

  // If position score is valid, project center3D to right image
  // Otherwise, keep 0, 0 value
  if( position.is_valid() )
  {
    position.center3d_proj_to_right_image = project_to_right_image( position.center3d );
  }

  // Debug print
  if( m_verbose )
  {
    print( position.center3d, "CENTER" );
    print( position.center3d_proj_to_right_image, "PROJ CENTER TO RIGHT" );
    print( position.rectified_left_bbox, "RECTIFIED LEFT BBOX" );
    print( position.left_bbox_proj_to_right_image, "PROJ LEFT BBOX" );
    std::cout << std::endl;
  }

  return position;
}

kwiver::vital::bounding_box_d
viame::pair_stereo_detections
::project_to_right_image( const kwiver::vital::bounding_box_d& bbox,
                          const kwiver::vital::image_of< float >& pos_3d_map ) const
{
  auto saturate_pos = [&pos_3d_map]( const kwiver::vital::vector_< 2, double >& corner )
  {
    auto x = std::min( std::max( corner.x(), 0. ), pos_3d_map.width() - 1. );
    auto y = std::min( std::max( corner.y(), 0. ), pos_3d_map.height() - 1. );

    return kwiver::vital::vector_< 2, double >{ x, y };
  };

  // Saturate upper left and lower right coordinates to image coordinates
  auto bbox_ul = saturate_pos( bbox.upper_left() );
  auto bbox_lr = saturate_pos( bbox.lower_right() );

  // Find 3D points associated with input bounding box
  auto const at = [ & ]( const kwiver::vital::vector_< 2, double >& corner )
  {
    auto const px = static_cast< size_t >( corner.x() );
    auto const py = static_cast< size_t >( corner.y() );

    return kv::vector_3d( pos_3d_map( px, py, 0 ), pos_3d_map( px, py, 1 ),
                          pos_3d_map( px, py, 2 ) );
  };

  auto const tl_3d = at( bbox_ul );
  auto const br_3d = at( bbox_lr );

  if( !point_is_valid( tl_3d ) || !point_is_valid( br_3d ) )
  {
    return {};
  }

  // Project points to right camera coordinates
  return project_to_right_image( std::vector< kv::vector_3d >{ tl_3d, br_3d } );
}

kwiver::vital::bounding_box_d
viame::pair_stereo_detections
::project_to_right_image( const std::vector< kwiver::vital::vector_3d >& points_3d ) const
{
  // Sanity check on input vect list
  if( points_3d.size() != 2 )
  {
    VITAL_THROW( kwiver::vital::invalid_data,
                 "Wrong input 3D point number. Expected 2, got : " + std::to_string( points_3d.size() ) );
  }

  // Project points to right camera coordinates
  auto const first = project_to_right_image( points_3d[ 0 ] );
  auto const second = project_to_right_image( points_3d[ 1 ] );

  return { first.x(), first.y(), second.x(), second.y() };
}

kwiver::vital::vector_2d
viame::pair_stereo_detections
::project_to_right_image( const kwiver::vital::vector_3d& points_3d ) const
{
  return mp::project_point( points_3d, m_R, m_T, m_K2, m_D2 );
}


std::vector< viame::Detections3DPositions >
viame::pair_stereo_detections
::update_left_detections_3d_positions(
    const std::vector< kwiver::vital::detected_object_sptr >& detections,
    const kwiver::vital::image& disparity_map ) const
{
  const auto pos_3d_map = reproject_3d_depth_map( disparity_map );
  std::vector< Detections3DPositions > positions;
  for( const auto& detection : detections )
  {
    positions.emplace_back( update_left_detection_3d_position( detection, pos_3d_map ) );
  }
  return positions;
}


viame::Detections3DPositions
viame::pair_stereo_detections
::update_left_detection_3d_position(
    const kwiver::vital::detected_object_sptr& detection,
    const kwiver::vital::image_of< float >& pos_3d_map ) const
{
  // Process 3D coordinates for frame matching the current depth image
  auto position = estimate_3d_position_from_detection( detection, pos_3d_map, true, 1.f / 3.f );

  // Add 3d estimations to state if score is valid
  if( position.score > 0 )
  {
    detection->add_note( ":stereo3d_x=" + std::to_string( position.center3d.x() ) );
    detection->add_note( ":stereo3d_y=" + std::to_string( position.center3d.y() ) );
    detection->add_note( ":stereo3d_z=" + std::to_string( position.center3d.z() ) );
    detection->add_note( ":score=" + std::to_string( position.score ) );
  }

  return position;
}


double
viame::pair_stereo_detections
::iou_distance( const kwiver::vital::bounding_box_d& bbox1,
                const kwiver::vital::bounding_box_d& bbox2 )
{
  kwiver::vital::aligned_box< double, 2 > bbox1_box{ bbox1.upper_left(), bbox1.lower_right() };
  kwiver::vital::aligned_box< double, 2 > bbox2_box{ bbox2.upper_left(), bbox2.lower_right() };

  // Early return if the input bounding boxes are invalid or don't intersect
  if( !bbox1.is_valid() || !bbox2.is_valid() || !bbox1_box.intersects( bbox2_box ) )
  {
    return 0;
  }

  auto bbox_intersection = bbox1_box.intersection( bbox2_box ).volume();
  auto bbox_union = bbox1_box.volume() + bbox2_box.volume() - bbox_intersection;
  return bbox_intersection / bbox_union;
}


kwiver::vital::image_of< float >
viame::pair_stereo_detections
::reproject_3d_depth_map( const kwiver::vital::image& disparity_left ) const
{
  // Every path that consumes the rectification transforms runs through here
  // first, so this is where a single-file calibration gets them derived.
  ensure_rectification( disparity_left.width(), disparity_left.height() );

  return mp::reproject_to_3d( disparity_left, m_Q );
}


void
viame::pair_stereo_detections
::ensure_rectification( size_t width, size_t height ) const
{
  if( m_rectified )
  {
    return;
  }

  auto const rectification = mp::stereo_rectify(
    m_K1, m_D1, m_K2, m_D2, width, height, m_R, m_T );

  m_R1 = rectification.left_rotation;
  m_R2 = rectification.right_rotation;
  m_P1 = rectification.left_projection;
  m_P2 = rectification.right_projection;
  m_Q = rectification.disparity_to_depth;

  m_rectified = true;
}


kwiver::vital::vector_2d
viame::pair_stereo_detections
::undistort_point( const kwiver::vital::vector_2d& point,
                   bool is_left_image ) const
{
  if( is_left_image )
  {
    return mp::undistort_point( point, m_K1, m_D1, m_R1, m_P1 );
  }

  return mp::undistort_point( point, m_K2, m_D2, m_R2, m_P2 );
}


std::vector< kwiver::vital::vector_2d >
viame::pair_stereo_detections
::undistort_point( const std::vector< kwiver::vital::vector_2d >& point,
                   bool is_left_image ) const
{
  std::vector< kwiver::vital::vector_2d > points_undist;
  points_undist.reserve( point.size() );

  for( auto const& one : point )
  {
    points_undist.push_back( undistort_point( one, is_left_image ) );
  }

  return points_undist;
}


/// @class ProcessTracker
/// @brief Helper class to track the processed detection during the different processing
template< typename T >
class ProcessTracker
{
public:
  bool is_processed( const T& value ) const
  {
    return m_processed.find( value ) != std::end( m_processed );
  }

  void emplace( const T& value )
  {
    m_processed.emplace( value );
  }

  void clear()
  {
    m_processed.clear();
  }
private:
  std::set< T > m_processed;
};

std::vector< std::vector< size_t > >
viame::pair_stereo_detections
::pair_left_right_detections_using_3d_center(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections )
{
  std::vector< std::vector< size_t > > paired_detections;
  ProcessTracker< size_t > tracker;

  const auto most_probable_right_detection = [&right_detections, &tracker](
      const kwiver::vital::vector_< 2, double >& left_point, const std::string& left_class )
  {
    int i_best = -1;
    auto dist_best = std::numeric_limits< double >::max();

    for( size_t i_right = 0; i_right < right_detections.size(); i_right++ )
    {
      // Skip right tracks not in current frame or with different detection class
      const auto& right_detection = right_detections[i_right];
      if( most_likely_detection_class( right_detection ) != left_class || tracker.is_processed( i_right ) )
      {
        continue;
      }

      auto right_bbox = right_detection->bounding_box();
      if( !right_bbox.is_valid() || !right_bbox.contains( left_point ) )
      {
        continue;
      }

      const auto dist = ( right_bbox.center() - left_point ).norm();
      if( dist < dist_best )
      {
        i_best = ( int ) i_right;
        dist_best = dist;
      }
    }
    return i_best;
  };

  for( size_t i_left = 0; i_left < left_detections.size(); i_left++ )
  {
    const auto& left_detection = left_detections[i_left];
    const auto left_class = most_likely_detection_class( left_detection );

    // Skip left tracks not in current frame or invalid
    if( !left_3d_pos[i_left].is_valid() )
    {
      continue;
    }

    // Find most probable right track match given projected center point
    const auto proj_left_point = left_3d_pos[i_left].center3d_proj_to_right_image;
    const auto i_right = most_probable_right_detection(
      kv::vector_2d( proj_left_point.x(), proj_left_point.y() ), left_class );
    if( i_right < 0 )
    {
      continue;
    }

    paired_detections.emplace_back( std::vector< size_t >{
      static_cast< size_t >( i_left ), static_cast< size_t >( i_right ) } );
    tracker.emplace( i_right );
  }
  return paired_detections;
}


std::vector< std::vector< size_t > >
viame::pair_stereo_detections
::pair_left_right_tracks_using_bbox_iou(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections,
    bool do_rectify_bbox )
{
  std::vector< std::vector< size_t > > paired_detections;
  ProcessTracker< size_t > tracker;

  const auto most_probable_right_track = [&right_detections, do_rectify_bbox, &tracker, this](
      const kwiver::vital::detected_object_sptr& left_detection, const std::string& left_class )
  {
    int i_best = -1;
    auto best_iou = std::numeric_limits< double >::lowest();
    auto left_bbox = left_detection->bounding_box();
    if( do_rectify_bbox )
    {
      left_bbox = get_rectified_bbox( left_bbox, true );
    }

    if( !left_bbox.is_valid() )
    {
      return i_best;
    }

    for( size_t i_right = 0; i_right < right_detections.size(); i_right++ )
    {
      // Skip right tracks not in current frame or with different detection class
      const auto& right_track = right_detections[i_right];
      if( most_likely_detection_class( right_track ) != left_class || tracker.is_processed( i_right ) )
      {
        continue;
      }

      auto right_bbox = right_track->bounding_box();
      if( !right_bbox.is_valid() )
      {
        continue;
      }

      if( do_rectify_bbox )
      {
        right_bbox = get_rectified_bbox( right_bbox, true );
      }

      const auto iou = iou_distance( left_bbox, right_bbox );
      if( ( iou > m_iou_pair_threshold ) && ( iou > best_iou ) )
      {
        i_best = ( int ) i_right;
        best_iou = iou;
      }
    }
    return i_best;
  };

  for( size_t i_left = 0; i_left < left_detections.size(); i_left++ )
  {
    const auto& left_detection = left_detections[i_left];
    const auto left_track_class = most_likely_detection_class( left_detection );

    // Find most probable right track match given projected center point
    const auto i_right = most_probable_right_track( left_detection, left_track_class );
    if( i_right < 0 )
    {
      continue;
    }

    paired_detections.emplace_back( std::vector< size_t >{
      static_cast< size_t >( i_left ), static_cast< size_t >( i_right ) } );
    tracker.emplace( i_right );
  }
  return paired_detections;
}


std::vector< std::vector< size_t > >
viame::pair_stereo_detections
::pair_left_right_detections(
    const std::vector< kwiver::vital::detected_object_sptr >& left_detections,
    const std::vector< viame::Detections3DPositions >& left_3d_pos,
    const std::vector< kwiver::vital::detected_object_sptr >& right_detections )
{
  bool do_rectify_bbox = m_pairing_method == "PAIRING_RECTIFIED_IOU";
  if( m_pairing_method == "PAIRING_3D" )
  {
    return pair_left_right_detections_using_3d_center( left_detections, left_3d_pos, right_detections );
  }
  else
  {
    return pair_left_right_tracks_using_bbox_iou( left_detections, right_detections, do_rectify_bbox );
  }
}


std::string
viame::pair_stereo_detections
::most_likely_detection_class( const kwiver::vital::detected_object_sptr& detection )
{
  if( !detection )
  {
    return {};
  }

  auto detection_type = detection->type();
  if( !detection_type )
  {
    return {};
  }

  std::string most_likely;
  detection_type->get_most_likely( most_likely );
  return most_likely;
}