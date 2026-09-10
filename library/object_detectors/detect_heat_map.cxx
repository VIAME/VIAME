// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of ocv::detect_heat_map

#include "detect_heat_map.h"

#include <viame/algorithm_framework/config/config_difference.h>
#include <viame/algorithm_framework/exceptions.h>
#include <viame/core_types/detected_object.h>
#include <viame/core_types/detected_object_type.h>
#include <viame/algorithm_framework/util/wall_timer.h>

#include <image_ops/contours.h>
#include <image_ops/filter.h>
#include <image_ops/histogram.h>
#include <image_ops/pixel.h>
#include <image_ops/warp.h>

#include <viame/core_types/image_container.h>


namespace kwiver {

namespace arrows {

namespace ocv {

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

using namespace kwiver::vital;

template < class T >
static
std::vector< T >
linspace( T a, T b, int n )
{
  if( n == 0 )
  {
    VITAL_THROW( invalid_value, "n must be a positive integer." );
  }

  std::vector< T > array;
  if( n == 1 )
  {
    array.push_back( a );
    return array;
  }

  double epsilon = 0.0001;
  double step = ( b - a ) / ( n - 1 );
  if( a == b )
  {
    for( int i = 0; i < n; i++ )
    {
      array.push_back( a );
    }
  }
  else if( step >= 0 )
  {
    while( a <= b + epsilon )
    {
      array.push_back( a );
      a += step;
    }
  }
  else
  {
    while( a + epsilon >= b )
    {
      array.push_back( a );
      a += step;
    }
  }
  return array;
}

// ----------------------------------------------------------------------------
///
/// @brief Applies threshold and finds bounding boxes for above-zero pixels.
///
/// @param image Image
/// @param threshold Threshold used to turn image into a binary max.
/// @param first_row First row from which to start checking for the start of the
///  bounding box.
/// @param last_row One greater than the index for the last row from which to
///  start checking for a viable bounding box (default -1 uses image height).
/// @param first_row First column from which to start checking for the start of
///  the bounding box.
/// @param last_row One greater than the index for the last column from which to
///  start checking for a viable bounding box (default -1 uses image width).
///
/// @return Tuple of integers (first row, last row, first col, last col)
///  indicating the bounding rows/columns where at least one above-threshold
///  element exists. last_row is one greater than the index for the last above-
///  threshold row, and last_col is one greater than the index for the last
///  above-threshold column. If image is entirely below threshold, then
///  first_row = last_row = image.rows and first_colum = last_column =
///  image.cols.
template < class T >
std::tuple< int, int, int, int >
static
mask_bounding_box(
  kv::image_of< T > const& image, double threshold = 0, int first_row = 0,
  int last_row = -1, int first_col = 0, int last_col = -1 )
{
  if( image.depth() > 1 )
  {
    VITAL_THROW( vital::invalid_data, "image must be single channel." );
  }

  auto const rows = static_cast< int >( image.height() );
  auto const cols = static_cast< int >( image.width() );

  // Find the first/last non-zero rows and columns where we should consider
  // centering a bounding box.
  if( last_row == -1 )
  {
    last_row = rows;
  }
  if( last_col == -1 )
  {
    last_col = cols;
  }

  --last_col;
  --last_row;

  auto const at = [ & ]( int row, int col )
  {
    return static_cast< double >(
      image( static_cast< size_t >( col ), static_cast< size_t >( row ), 0 ) );
  };

  bool done = false;
  while( first_row < rows )
  {
    for( int j = 0; j < cols; j++ )
    {
      if( at( first_row, j ) >= threshold )
      {
        done = true;
        break;
      }
    }
    if( done )
    {
      break;
    }
    ++first_row;
  }

  done = false;
  while( last_row > first_row )
  {
    for( int j = 0; j < cols; j++ )
    {
      if( at( last_row, j ) >= threshold )
      {
        done = true;
        break;
      }
    }
    if( done )
    {
      break;
    }
    --last_row;
  }

  done = false;
  while( first_col < cols )
  {
    for( int i = 0; i < rows; i++ )
    {
      if( at( i, first_col ) >= threshold )
      {
        done = true;
        break;
      }
    }
    if( done )
    {
      break;
    }
    ++first_col;
  }

  done = false;
  while( last_col > first_col )
  {
    for( int i = 0; i < rows; i++ )
    {
      if( at( i, last_col ) >= threshold )
      {
        done = true;
        break;
      }
    }
    if( done )
    {
      break;
    }
    --last_col;
  }

  return std::make_tuple( first_row, last_row + 1, first_col, last_col + 1 );
}

// ----------------------------------------------------------------------------
// ----------------------------- Sprokit --------------------------------------

/// Private implementation class
class detect_heat_map::priv
{
public:
  double
  m_threshold() const { return parent.get_threshold(); }
  int
  m_force_bbox_width() const { return parent.get_force_bbox_width(); }
  int
  m_force_bbox_height() const { return parent.get_force_bbox_height(); }
  int
  m_bbox_buffer() const { return parent.get_bbox_buffer(); }
  int
  m_min_area() const { return parent.get_min_area(); }
  int
  m_max_area() const { return parent.get_max_area(); }
  double
  m_min_fill_fraction() const { return parent.get_min_fill_fraction(); }
  std::string
  m_class_name() const { return parent.get_class_name(); }
  std::string
  m_score_mode() const { return parent.get_score_mode(); }
  int
  m_max_boxes() const { return parent.get_max_boxes(); }
  int
  m_pyr_red_levels() const { return parent.get_pyr_red_levels(); }

  double
  m_fixed_score() const
  {
    // Extract a numerical score from the score_mode string if possible.
    char* p;
    double converted = strtod( this->m_score_mode().c_str(), &p );
    if( !( *p ) )
    {
      // d_->m_score_mode = "fixed"; // we keep the value unchanged so we can
      // report it in get_configuration where m_fixed_score does not exist
      return converted;
    }
    return -1;
  }

  // ----
  // Only ever assigned true, in check_configuration(). Left uninitialised, a
  // detector configured for connected components (force_bbox_* = -1) can take
  // the fixed-size branch and build a box filter from those -1 dimensions.
  bool m_force_bbox_size = false;
  kwiver::vital::logger_handle_t m_logger;
  kwiver::vital::wall_timer m_timer;

  detect_heat_map& parent;

  /// Constructor
  priv( detect_heat_map& parent )
    : parent( parent )
  {}

  // --------------------------------------------------------------------------
  detected_object_set_sptr
  get_bounding_boxes( kv::image_of< uint8_t > const& heat_map )
  {
    if( m_force_bbox_size )
    {
      if( m_threshold() != -1 )
      {
        return get_bbox_fixed_size( threshold_binary( heat_map,
                                                      m_threshold() ) );
      }
      else
      {
        return get_bbox_fixed_size( heat_map );
      }
    }
    else
    {
      return get_bbox_ccomponents( heat_map );
    }
  }

  // --------------------------------------------------------------------------
  /// `cv::threshold` with `THRESH_BINARY` and a max value of one.
  ///
  /// Strictly greater than, which is what OpenCV's THRESH_BINARY is: a pixel
  /// equal to the threshold is background. That is why a `threshold` of 0
  /// keeps only the non-zero pixels rather than everything.
  static kv::image_of< uint8_t >
  threshold_binary( kv::image_of< uint8_t > const& image, double level )
  {
    kv::image_of< uint8_t > out( image.width(), image.height(), 1 );

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        out( i, j, 0 ) =
          ( static_cast< double >( image( i, j, 0 ) ) > level ) ? 1 : 0;
      }
    }

    return out;
  }

  // --------------------------------------------------------------------------
  /// Place fixed-size windows greedily, brightest first.
  ///
  /// `get_bbox_fixed_size_dense` was beside this one and nothing called it;
  /// P7-T04 deleted it rather than porting two hundred lines of dead code.
  detected_object_set_sptr
  get_bbox_fixed_size( kv::image_of< uint8_t > const& heat_map0 )
  {
    int bbox_height = m_force_bbox_height();
    int bbox_width = m_force_bbox_width();
    int bbox_buffer_w = m_bbox_buffer();
    int bbox_buffer_h = m_bbox_buffer();

    LOG_TRACE(
      m_logger, "Creating bounding boxes of fixed size (" <<
        std::to_string( bbox_width ) << " x " <<
        std::to_string( bbox_height ) << ")" );

    if( static_cast< int >( heat_map0.height() ) < bbox_height ||
        static_cast< int >( heat_map0.width() ) < bbox_width )
    {
      VITAL_THROW(
        invalid_value, std::string( "Forced bounding box size exceeds " ) +
        "provided image size (" +
        std::to_string( heat_map0.width() ) + " x " +
        std::to_string( heat_map0.height() ) + ")" );
    }

    double bbox_out_width_rescale = 1;
    double bbox_out_height_rescale = 1;

    kv::image_of< uint8_t > heat_map;

    m_timer.start();

    // Reduce heat map by 2^pyr_levels and consider coarser placement of bboxes
    if( m_pyr_red_levels() > 0 )
    {
      heat_map = io::normalize_min_max( heat_map0, 0.0, 255.0 );

      for( int i = 0; i < m_pyr_red_levels(); ++i )
      {
        heat_map = pyr_down( heat_map );
      }

      // Integer division, as it was: the scale is a whole number or it is
      // one, and a 15 pixel image reduced once scales by 1 rather than by
      // 1.875
      double scale_width =
        static_cast< int >( heat_map0.width() ) /
        static_cast< int >( heat_map.width() );
      double scale_height =
        static_cast< int >( heat_map0.height() ) /
        static_cast< int >( heat_map.height() );

      bbox_out_width_rescale = scale_width;
      bbox_out_height_rescale = scale_height;
      bbox_height /= scale_height;
      bbox_width /= scale_width;
      bbox_buffer_w /= scale_width;
      bbox_buffer_h /= scale_height;
    }
    else
    {
      // A deep copy: the search below erases each box it takes, and
      // `vital::image` copies share their memory
      heat_map = kv::image_of< uint8_t >( heat_map0.width(),
                                          heat_map0.height(), 1 );

      for( size_t j = 0; j < heat_map0.height(); ++j )
      {
        for( size_t i = 0; i < heat_map0.width(); ++i )
        {
          heat_map( i, j, 0 ) = heat_map0( i, j, 0 );
        }
      }
    }

    m_timer.stop();
    LOG_DEBUG(
      m_logger,
      "Image pyramiding elapsed time: " << m_timer.elapsed() );

    int hmap_w = static_cast< int >( heat_map.width() );
    int hmap_h = static_cast< int >( heat_map.height() );

    // For a bounding box 'centered' on pixel indices (x,y), the upper left
    // corner coordinates will be (x-hr1f, y-vr1f), and the lower right corner
    // will have coordinates (x+hr2f, y+vr2f) inclusive.
    int hr1f = bbox_width / 2;
    int vr1f = bbox_height / 2;
    int hr2f = bbox_width - 1 - hr1f;
    int vr2f = bbox_height - 1 - vr1f;

    // Width and height of the reduced-size one that is used to accommodate
    // bbox_buffer. For this reduced version of the bounding box 'centered' on
    // pixel indices (x,y), the upper left corner coordinates will be
    // (x-hr1, y-vr1), and the lower right corner will have coordinates
    // (x+hr2, y+vr2)
    int bbox_w_red = bbox_width - bbox_buffer_w * 2;
    int bbox_h_red = bbox_height - bbox_buffer_h * 2;
    LOG_TRACE( m_logger, "kernel size: " << bbox_w_red << " x " << bbox_h_red );

    // The mean over the reduced box, which is what `cv::boxFilter` with
    // `normalize` computes. The anchor is the kernel's centre, and
    // `filter_2d` puts it at `size / 2` -- the same "round the centre up"
    // that the original spelled out for an even kernel.
    //
    // BORDER_CONSTANT, so a box hanging off the edge is averaged against
    // zeros and scores lower than one wholly inside. That is what makes the
    // placement pull away from the border.
    io::kernel box;
    box.width = static_cast< size_t >( bbox_w_red );
    box.height = static_cast< size_t >( bbox_h_red );
    box.weights.assign(
      box.width * box.height,
      1.0 / static_cast< double >( box.width * box.height ) );

    auto detected_objects = std::make_shared< detected_object_set >();
    int x1, x2, y1, y2, x1t, x2t, y1t, y2t, dx, dy;
    int cntr = 0;

    while( true )
    {
      // We are searching locations to place the bounding boxes that maximize
      // the enclosed sum value from heat_map. Therefore, we can use a box
      // blur filter to efficiently calculate this. The kernel size is equal
      // to the bounding box size minus the bounding box buffer. This way, we
      // only consider the inner useful region of the bounding box when
      // looking for optimal placement.
      auto const conv_map = io::filter_2d< float >(
        heat_map, box, io::border_mode::CONSTANT );

      io::extremum lowest;
      io::extremum highest;
      io::min_max( conv_map, lowest, highest );

      auto const max_val = highest.value;

      if( max_val == 0 )
      {
        // No above-threshold regions left.
        break;
      }

      auto max_x = static_cast< int >( highest.i );
      auto max_y = static_cast< int >( highest.j );

      // Define the bounding box
      // vital::bounding_box lower-right point is not inclusive, so must add 1.
      x1 = max_x - hr1f;
      y1 = max_y - vr1f;
      x2 = max_x + hr2f + 1;
      y2 = max_y + vr2f + 1;
      dx = -std::min( 0, x1 ) - std::max( 0, x2 - hmap_w );
      dy = -std::min( 0, y1 ) - std::max( 0, y2 - hmap_h );
      x1 += dx;
      x2 += dx;
      y1 += dy;
      y2 += dy;

      // In cases where there are many different positions where the bounding
      // box to still covers all elements, the above approach picks the first
      // one found, which is often not ideal. Ideally, the enclosed elements
      // would be centered in the bounding box.
      std::tie( y1t, y2t, x1t, x2t ) = mask_bounding_box< uint8_t >(
        heat_map,
        1, y1, y2,
        x1, x2 );

      if( x2t > x1t )
      {
        max_x = ( x1t + x2t ) / 2;
      }
      if( y2t > y1t )
      {
        max_y = ( y1t + y2t ) / 2;
      }

      // vital::bounding_box lower-right point is not inclusive, so must add 1.
      y1 = max_y - vr1f;
      y2 = max_y + vr2f + 1;
      x1 = max_x - hr1f;
      x2 = max_x + hr2f + 1;

      // Reposition, if necessary, so that the bounding box is completely
      // within the image.
      dx = -std::min( 0, x1 ) - std::max( 0, x2 - hmap_w );
      dy = -std::min( 0, y1 ) - std::max( 0, y2 - hmap_h );
      x1 += dx;
      x2 += dx;
      y1 += dy;
      y2 += dy;

      kwiver::vital::bounding_box_d bbox( x1 * bbox_out_width_rescale,
        y1 * bbox_out_height_rescale,
        x2 * bbox_out_width_rescale,
        y2 * bbox_out_height_rescale );

      LOG_TRACE(
        m_logger, "Creating bounding box (" <<
          std::to_string( bbox.min_x() ) << ", " <<
          std::to_string( bbox.max_x() ) << ", " <<
          std::to_string( bbox.min_y() ) << ", " <<
          std::to_string( bbox.max_y() ) << ")" );

      auto dot = std::make_shared< detected_object_type >();
      dot->set_score( m_class_name(), max_val );
      detected_objects->add(
        std::make_shared< kwiver::vital::detected_object >(
          bbox, max_val, dot ) );

      // Erase the region so the next iteration looks elsewhere.
      for( int j = y1; j < y2; ++j )
      {
        for( int i = x1; i < x2; ++i )
        {
          heat_map( static_cast< size_t >( i ),
                    static_cast< size_t >( j ), 0 ) = 0;
        }
      }

      ++cntr;
      if( cntr == m_max_boxes() )
      {
        break;
      }
    }

    return detected_objects;
  }

  // --------------------------------------------------------------------------
  /// Halve the image, which is `cv::pyrDown`.
  ///
  /// A five by five Gaussian and then every second pixel, and the kernel is
  /// OpenCV's own: (1, 4, 6, 4, 1) / 16 in each axis. The output is
  /// `(width + 1) / 2` by `(height + 1) / 2`, again as OpenCV's is.
  static kv::image_of< uint8_t >
  pyr_down( kv::image_of< uint8_t > const& image )
  {
    std::vector< double > const line{ 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0,
                                      4.0 / 16.0, 1.0 / 16.0 };

    auto const blurred = io::filter_2d(
      image, io::separable_kernel( line, line ),
      io::border_mode::REFLECT_101 );

    auto const width = ( image.width() + 1 ) / 2;
    auto const height = ( image.height() + 1 ) / 2;

    kv::image_of< uint8_t > out( width, height, image.depth() );

    for( size_t plane = 0; plane < image.depth(); ++plane )
    {
      for( size_t j = 0; j < height; ++j )
      {
        for( size_t i = 0; i < width; ++i )
        {
          out( i, j, plane ) = blurred( i * 2, j * 2, plane );
        }
      }
    }

    return out;
  }

  // --------------------------------------------------------------------------
  /// Threshold image and find connected components of binary image
  detected_object_set_sptr
  get_bbox_ccomponents( kv::image_of< uint8_t > const& heat_map )
  {
    auto mask = threshold_binary( heat_map, m_threshold() );

    auto detected_objects = std::make_shared< detected_object_set >();

    // Remove outer border of pixels because findContours has trouble with
    // regions connected to the edge of the image.
    for( size_t i = 0; i < mask.width(); ++i )
    {
      mask( i, 0, 0 ) = 0;
      mask( i, mask.height() - 1, 0 ) = 0;
    }

    for( size_t j = 0; j < mask.height(); ++j )
    {
      mask( 0, j, 0 ) = 0;
      mask( mask.width() - 1, j, 0 ) = 0;
    }

    auto const contours = io::find_contours( mask );

    auto dot = std::make_shared< detected_object_type >();
    dot->set_score( m_class_name(), m_fixed_score() );

    for( auto const& contour : contours )
    {
      // Note that this area is the polygon through the pixel *centres*, not
      // the count of pixels inside it -- `cv::contourArea` answers the same
      // and `min_area` has always been read against that.
      double const area = io::contour_area( contour );

      if( area >= m_min_area() && area <= m_max_area() )
      {
        auto const bounds = io::bounding_rect( contour );

        if( area >= static_cast< double >( bounds.width() ) *
                    static_cast< double >( bounds.height() ) *
                    m_min_fill_fraction() )
        {
          kwiver::vital::bounding_box_d bbox( bounds.left, bounds.top,
                                              bounds.right, bounds.bottom );

          detected_objects->add(
            std::make_shared< kwiver::vital::detected_object >(
              bbox,
              m_fixed_score(),
              dot ) );
        }
      }
    }

    LOG_TRACE( m_logger, "Finished creating bounding boxes" );
    return detected_objects;
  }

  // --------------------------------------------------------------------------
};

void
detect_heat_map
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.ocv.detect_heat_map" );
  d_->m_logger = logger();
}

/// Destructor
detect_heat_map
::~detect_heat_map() noexcept
{}

/// Set this algo's properties via a config block
void
detect_heat_map
::set_configuration_internal(
  [[maybe_unused]] vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, in_config );
  cd.warn_extra_keys( logger() );

  if( ( d_->m_force_bbox_width() == -1  && d_->m_force_bbox_height() != -1 ) ||
      ( d_->m_force_bbox_width() != -1  && d_->m_force_bbox_height() == -1 ) ||
      ( d_->m_force_bbox_width() != -1  && d_->m_force_bbox_width() <= 0 )   ||
      ( d_->m_force_bbox_height() != -1 && d_->m_force_bbox_height() <= 0 ) )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "'force_bbox_width' and "
      "'force_bbox_height' must both be "
      "-1, indicating that a particular "
      "bounding box size will not be "
      "enforced, or both positive, "
      "indicating the size of the "
      "bounding box that will be "
      "enforced." );
  }

  if( d_->m_force_bbox_width() > 0  && d_->m_force_bbox_height() > 0 )
  {
    if( d_->m_force_bbox_width() - d_->m_bbox_buffer() <= 0 )
    {
      VITAL_THROW(
        algorithm_configuration_exception, interface_name(), impl_name(),
        "(force_bbox_width - bbox_buffer) "
        "must be positive." );
    }

    if( d_->m_force_bbox_height() - d_->m_bbox_buffer() <= 0 )
    {
      VITAL_THROW(
        algorithm_configuration_exception, interface_name(), impl_name(),
        "(force_bbox_height - "
        "bbox_buffer) must be "
        "positive." );
    }

    d_->m_force_bbox_size = true;
  }
  else if( d_->m_threshold() == -1 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "If 'force_bbox_width' and "
      "'force_bbox_height' are not set,"
      "then a positive 'threshold' is "
      "required." );
  }

  if( d_->m_threshold() < 0 && d_->m_threshold() != -1 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "'threshold' must be non-negative "
      "in order for valid thresholding "
      "or equal to -1, indicating that "
      "no thresholding will be done." );
  }

  LOG_DEBUG( logger(), "threshold: " << std::to_string( d_->m_threshold() ) );
  LOG_DEBUG(
    logger(),
    "force_bbox_width: " << std::to_string( d_->m_force_bbox_width() ) );
  LOG_DEBUG(
    logger(),
    "force_bbox_height: " << std::to_string( d_->m_force_bbox_height() ) );
  LOG_DEBUG(
    logger(),
    "bbox_buffer: " << std::to_string( d_->m_bbox_buffer() ) );
  LOG_DEBUG( logger(), "min_area: " << std::to_string( d_->m_min_area() ) );
  LOG_DEBUG( logger(), "max_area: " << std::to_string( d_->m_max_area() ) );
  LOG_DEBUG(
    logger(),
    "min_fill_fraction: " << std::to_string( d_->m_min_fill_fraction() ) );
  LOG_DEBUG( logger(), "class_name: " << d_->m_class_name() );

  LOG_DEBUG( logger(), "score_mode: " << d_->m_score_mode() );
  LOG_DEBUG( logger(), "fixed_score: " << d_->m_fixed_score() );
}

bool
detect_heat_map
::check_configuration( vital::config_block_sptr config_in ) const
{
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, config_in );
  return !cd.warn_extra_keys( logger() );
}

/// Return homography to stabilize the image_src relative to the key frame
detected_object_set_sptr
detect_heat_map
::detect( image_container_sptr image_data ) const
{
  if( !image_data )
  {
    VITAL_THROW(
      vital::invalid_data,
      "Inputs to ocv::detect_heat_map are null" );
  }
  LOG_TRACE( logger(), "Received image" );

  auto const source = image_data->get_image();

  if( source.depth() > 1 )
  {
    VITAL_THROW(
      vital::invalid_data,
      "Heat map image must be single channel." );
  }

  return d_->get_bounding_boxes( kv::image_of< uint8_t >( source ) );
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver
