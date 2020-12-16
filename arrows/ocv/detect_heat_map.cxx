// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of ocv::detect_heat_map
 */

#include "detect_heat_map.h"

#include <vital/exceptions.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/util/wall_timer.h>
#include <vital/config/config_difference.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

using namespace kwiver::vital;

template <class T>
static
std::vector<T>
linspace(T a, T b, int n)
{
  if( n == 0 )
  {
    VITAL_THROW( invalid_value, "n must be a positive integer.");
  }

  std::vector<T> array;
  if( n == 1 )
  {
    array.push_back(a);
    return array;
  }
  double epsilon = 0.0001;
  double step = (b-a) / (n-1);
  if (a==b)
  {
    for (int i = 0; i < n; i++)
    {
      array.push_back(a);
    }
  }
  else if (step >= 0)
  {
    while(a <= b + epsilon)
    {
        array.push_back(a);
        a += step;
    }
  }
  else
  {
    while(a + epsilon >= b )
    {
        array.push_back(a);
        a += step;
    }
  }
  return array;
}

//-----------------------------------------------------------------------------
  ///
  /**
  * @brief Applies threshold and finds bounding boxes for above-zero pixels.
  *
  * @param image Image
  * @param threshold Threshold used to turn image into a binary max.
  * @param first_row First row from which to start checking for the start of the
  *   bounding box.
  * @param last_row One greater than the index for the last row from which to
  *   start checking for a viable bounding box (default -1 uses image height).
  * @param first_row First column from which to start checking for the start of
  *   the bounding box.
  * @param last_row One greater than the index for the last column from which to
  *   start checking for a viable bounding box (default -1 uses image width).
  *
  * @return Tuple of integers (first row, last row, first col, last col)
  *   indicating the bounding rows/columns where at least one above-threshold
  *   element exists. last_row is one greater than the index for the last above-
  *   threshold row, and last_col is one greater than the index for the last
  *   above-threshold column. If image is entirely below threshold, then
  *   first_row = last_row = image.rows and first_colum = last_column =
  *   image.cols.
  */
template <class T>
std::tuple<int,int,int,int>
static
mask_bounding_box( const cv::Mat image, double threshold = 0, int first_row = 0,
                   int last_row = -1, int first_col = 0, int last_col = -1 )
{
  if ( image.channels() > 1 )
  {
    VITAL_THROW( vital::invalid_data, "image must be single channel.");
  }

  // Find the first/last non-zero rows and columns where we should consider
  // centering a bounding box.
  if( last_row == -1 )
  {
    last_row = image.rows;
  }
    if( last_col == -1 )
  {
    last_col = image.cols;
  }

  --last_col;
  --last_row;

  bool done = false;
  while( first_row < image.rows )
  {
    for( int j=0; j < image.cols; j++ )
    {
      if( image.at<T>( first_row, j ) >= threshold )
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
    for( int j=0; j < image.cols; j++ )
    {
      if( image.at<T>( last_row, j ) >= threshold )
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
  while( first_col < image.cols )
  {
    for( int i=0; i<image.rows; i++ )
    {
      if( image.at<T>( i, first_col ) >= threshold )
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
    for( int i=0; i < image.rows; i++ )
    {
      if( image.at<T>( i, last_col ) >= threshold )
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

  return std::make_tuple( first_row, last_row+1, first_col, last_col+1 );
}

// ---------------------------------------------------------------------------
// ------------------------------ Sprokit ------------------------------------

/// Private implementation class
class detect_heat_map::priv
{
public:
  double m_threshold;
  int m_force_bbox_width, m_force_bbox_height;
  bool m_force_bbox_size;
  int m_bbox_buffer;
  int m_min_area, m_max_area;
  double m_min_fill_fraction;
  double m_fixed_score;
  std::string m_class_name, m_score_mode;
  int m_max_boxes, m_pyr_red_levels;
  kwiver::vital::logger_handle_t m_logger;
  kwiver::vital::wall_timer m_timer;

  /// Constructor
  priv()
    :
      m_threshold(-1),
      m_force_bbox_width(-1),
      m_force_bbox_height(-1),
      m_force_bbox_size(false),
      m_bbox_buffer(0),
      m_min_area(1),
      m_max_area(10000000),
      m_min_fill_fraction(0.25),
      m_fixed_score(-1),
      m_class_name("unspecified"),
      m_score_mode("1"),
      m_max_boxes(1000000),
      m_pyr_red_levels(0)
  {
  }

  // --------------------------------------------------------------------------
  detected_object_set_sptr
  get_bounding_boxes(cv::Mat const &heat_map)
  {
    if( m_force_bbox_size )
    {
      if( m_threshold != -1 )
      {
        cv::Mat mask;
        cv::threshold( heat_map, mask, m_threshold, 1, cv::THRESH_BINARY );
        return get_bbox_fixed_size( mask );
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
  /// Consider windows on a dense, fixed grid, removing empty ones.
  detected_object_set_sptr
  get_bbox_fixed_size_dense( cv::Mat const &heat_map )
  {
    int hmap_w = heat_map.cols;
    int hmap_h = heat_map.rows;

    LOG_TRACE( m_logger, "Creating bounding boxes of fixed size (" <<
               std::to_string(m_force_bbox_width) << " x " <<
               std::to_string(m_force_bbox_height) << ")" );

    if( heat_map.rows < m_force_bbox_height ||
        heat_map.cols < m_force_bbox_width )
    {
      VITAL_THROW( invalid_value, std::string("Forced bounding box size exceeds") +
                           "provided image size (" + std::to_string(hmap_w) +
                           " x " + std::to_string(hmap_h) + ")" );
    }

    // For a bounding box 'centered' on pixel indices (x,y), the upper left
    // corner coordinates will be (x-hr1f, y-vr1f), and the lower right corner
    // will have coordinates (x+hr2f, y+vr2f) inclusive.
    int hr1f = m_force_bbox_width/2;
    int vr1f = m_force_bbox_height/2;
    int hr2f = m_force_bbox_width-1-hr1f;
    int vr2f = m_force_bbox_height-1-vr1f;

    // Width and height of the reduced-size one that is used to accommodate
    // bbox_buffer. For this reduced version of the bounding box 'centered' on
    // pixel indices (x,y), the upper left corner coordinates will be
    // (x-hr1, y-vr1), and the lower right corner will have coordinates
    // (x+hr2, y+vr2)
    int bbox_w_red = m_force_bbox_width-m_bbox_buffer;
    int bbox_h_red = m_force_bbox_height-m_bbox_buffer;
    int hr1 = bbox_w_red/2;
    int vr1 = bbox_h_red/2;
    //int hr2 = bbox_w_red-1-hr1;
    //int vr2 = bbox_h_red-1-vr1;
    cv::Size ksize( bbox_w_red, bbox_h_red );

    // When the kernel size is even, the center is ambiguous. Let's explicitly
    // round the center up to the nearest integer.
    cv::Point anchor( hr1, vr1 );

    // We are searching locations to place the bounding boxes that maximize the
    // enclosed sum value from heat_map. Therefore, we can use a box blur filter
    // to efficiently calculate this. The kernel size is equal to the bounding
    // box size minus the bounding box buffer. This way, we only consider the
    // inner useful region of the bounding box when looking for optimal
    // placement. The value of the filtered image is equal to the sum of the
    // heat map value within a bounding box centered at that pixel.
    cv::Mat conv_map;
    cv::boxFilter( heat_map, conv_map, CV_32F, ksize, anchor, true,
                   cv::BORDER_CONSTANT );

    // The mask will indicate potential locations on which to center bounding
    // boxes.
    cv::Mat mask;
    cv::threshold( heat_map, mask, 0, 1, cv::THRESH_BINARY );
    mask.convertTo( mask, CV_8U );

    int first_row, last_row, first_col, last_col;
    std::tie( first_row, last_row, first_col, last_col ) = mask_bounding_box<uchar>( mask );

    // All 'on' locations within the heat map live inside a bound box with:
    int num_on_rows = (last_row - first_row);
    int num_on_cols = (last_col - first_col);

    // last_row and last_col are actually one greater than the index for the
    // last row/col.
    --last_row;
    --last_col;

    // Require this many bounding boxes to cover all 'on'Z pixels
    int num_bboxes_high = (num_on_rows+bbox_h_red-1)/bbox_h_red;
    int num_bboxes_wide = (num_on_cols+bbox_w_red-1)/bbox_w_red;

    // The above looked at non-zero pixels, but you can more conservatively hit
    // those pixels with a bbox completely contained within the image.
    first_row += vr1f;
    last_row -= vr2f;
    first_col += hr1f;
    last_col -= hr2f;

    if( last_row < first_row )
    {
      // Only one bounding box will fit vertically
      first_row = last_row;
    }

    if( last_col < first_col )
    {
      // Only one bounding box will fit horizontally
      first_col = last_col;
    }

    // col_inds and row_inds represent the center coordinate for each bounding
    // box.
    std::vector<int> col_inds = linspace( first_col, last_col, num_bboxes_wide );
    std::vector<int> row_inds = linspace( first_row, last_row, num_bboxes_high );

    auto detected_objects = std::make_shared< detected_object_set >();
    for( int row : row_inds )
    {
      for( int col : col_inds )
      {
        // vital::bounding_box lower-right point is not inclusive, so must add 1.
        kwiver::vital::bounding_box_d bbox( col-hr1f, row-vr1f,
                                            col+hr2f+1, row+vr2f+1 );

        auto val = conv_map.at<float>(row,col);
        if( val > 0 )
        {
          LOG_TRACE( m_logger, "Creating bounding box (" <<
                     std::to_string(bbox.min_x()) << ", " <<
                     std::to_string(bbox.max_x()) << ", " <<
                     std::to_string(bbox.min_y()) << ", " <<
                     std::to_string(bbox.max_y()) << ")" );

          auto dot = std::make_shared< detected_object_type >();
          dot->set_score( m_class_name, val );
          detected_objects->add( std::make_shared< kwiver::vital::detected_object >( bbox, val, dot ) );
        }
      }
    }

    return detected_objects;
  }

  // --------------------------------------------------------------------------
  /// Find optimal tiling of bounding boxes with fixed size
  detected_object_set_sptr
  get_bbox_fixed_size( cv::Mat const &heat_map0 )
  {
    cv::Mat heat_map;
    int bbox_height = m_force_bbox_height;
    int bbox_width = m_force_bbox_width;
    int bbox_buffer_w = m_bbox_buffer;
    int bbox_buffer_h = m_bbox_buffer;

    LOG_TRACE( m_logger, "Creating bounding boxes of fixed size (" <<
             std::to_string(bbox_width) << " x " <<
             std::to_string(bbox_height) << ")" );

    if( heat_map0.rows < bbox_height ||
        heat_map0.cols < bbox_width )
    {
      VITAL_THROW( invalid_value, std::string("Forced bounding box size exceeds ") +
                           "provided image size (" +
                           std::to_string(heat_map0.cols) + " x " +
                           std::to_string(heat_map0.rows) + ")" );
    }

    double bbox_out_width_rescale = 1;
    double bbox_out_height_rescale = 1;

    m_timer.start();
    // Reduce heat map by 2^pyr_levels and consider coarser placement of bboxes
    if( m_pyr_red_levels > 0 )
    {
      cv::normalize( heat_map0, heat_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      for( int i = 0; i < m_pyr_red_levels; ++i )
      {
        cv::pyrDown( heat_map, heat_map );
      }
      double scale_width = heat_map0.cols/heat_map.cols;
      double scale_height = heat_map0.rows/heat_map.rows;
      bbox_out_width_rescale = scale_width;
      bbox_out_height_rescale = scale_height;
      bbox_height /= scale_height;
      bbox_width /= scale_width;
      bbox_buffer_w /= scale_width;
      bbox_buffer_h /= scale_height;
    }
    else
    {
      heat_map = heat_map0;
    }

    //heat_map0.copyTo( heat_map0, heat_map );
    m_timer.stop();
    LOG_DEBUG( m_logger, "Image pyramiding elapsed time: " << m_timer.elapsed() );

    int hmap_w = heat_map.cols;
    int hmap_h = heat_map.rows;

    // For a bounding box 'centered' on pixel indices (x,y), the upper left
    // corner coordinates will be (x-hr1f, y-vr1f), and the lower right corner
    // will have coordinates (x+hr2f, y+vr2f) inclusive.
    int hr1f = bbox_width/2;
    int vr1f = bbox_height/2;
    int hr2f = bbox_width-1-hr1f;
    int vr2f = bbox_height-1-vr1f;

    // Width and height of the reduced-size one that is used to accommodate
    // bbox_buffer. For this reduced version of the bounding box 'centered' on
    // pixel indices (x,y), the upper left corner coordinates will be
    // (x-hr1, y-vr1), and the lower right corner will have coordinates
    // (x+hr2, y+vr2)
    int bbox_w_red = bbox_width-bbox_buffer_w*2;
    int bbox_h_red = bbox_height-bbox_buffer_h*2;
    int hr1 = bbox_w_red/2;
    int vr1 = bbox_h_red/2;
    //int hr2 = bbox_w_red-1-hr1;
    //int vr2 = bbox_h_red-1-vr1;
    LOG_TRACE( m_logger, "kernel size: " << bbox_w_red << " x " << bbox_h_red );
    cv::Size ksize( bbox_w_red, bbox_h_red );

    // When the kernel size is even, the center is ambiguous. Let's explicitly
    // round the center up to the nearest integer.
    cv::Point anchor( hr1, vr1 );

    cv::Mat conv_map;
    double min_val, max_val;
    cv::Point max_loc;
    auto detected_objects = std::make_shared< detected_object_set >();
    int x1, x2, y1, y2, x1t, x2t, y1t, y2t, dx, dy;
    int cntr = 0;
    while( true )
    {
      // We are searching locations to place the bounding boxes that maximize the
      // enclosed sum value from heat_map. Therefore, we can use a box blur filter
      // to efficiently calculate this. The kernel size is equal to the bounding
      // box size minus the bounding box buffer. This way, we only consider the
      // inner useful region of the bounding box when looking for optimal
      // placement. The value of the filtered image is equal to the sum of the
      // heat map value within a bounding box centered at that pixel.
      cv::boxFilter( heat_map, conv_map, CV_32F, ksize, anchor, true,
                     cv::BORDER_CONSTANT );

      if( false )
      {
        if( cntr == 0 )
        {
          cv::Mat output;
          cv::normalize( conv_map, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
          cv::imwrite( "/home/mattb/debug_output/output.tif", output );
          cv::normalize( heat_map, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
          cv::imwrite( "/home/mattb/debug_output/latest_heat_map.tif", output );
        }
      }

      cv::minMaxLoc(conv_map, &min_val, &max_val, NULL, &max_loc);
      if( max_val == 0 )
      {
        // No above-threshold regions left.
        break;
      }

      // Define the bounding box
      // vital::bounding_box lower-right point is not inclusive, so must add 1.
      x1 = max_loc.x-hr1f;
      y1 = max_loc.y-vr1f;
      x2 = max_loc.x+hr2f+1;
      y2 = max_loc.y+vr2f+1;
      dx = -std::min( 0, x1 ) - std::max( 0, x2 - hmap_w );
      dy = -std::min( 0, y1 ) - std::max( 0, y2 - hmap_h );
      x1 += dx;  x2 += dx;  y1 += dy;  y2 += dy;

      // In cases where there are many different positions where the bounding
      // box to still covers all elements, the above approach picks the first
      // one found, which is often not ideal. Ideally, the enclosed elements
      // would be centered in the bounding box.
      std::tie( y1t, y2t, x1t, x2t ) = mask_bounding_box<uchar>( heat_map,
                                                                 1, y1, y2,
                                                                 x1, x2 );

      if( x2t > x1t )
      {
        max_loc.x = (x1t+x2t)/2;
      }
      if( y2t > y1t )
      {
        max_loc.y = (y1t+y2t)/2;
      }

      // vital::bounding_box lower-right point is not inclusive, so must add 1.
      y1 = max_loc.y-vr1f;
      y2 = max_loc.y+vr2f+1;
      x1 = max_loc.x-hr1f;
      x2 = max_loc.x+hr2f+1;

      // Reposition, if necessary, so that the bounding box is completely
      // within the image.
      dx = -std::min( 0, x1 ) - std::max( 0, x2 - hmap_w );
      dy = -std::min( 0, y1 ) - std::max( 0, y2 - hmap_h );
      x1 += dx;  x2 += dx;  y1 += dy;  y2 += dy;

      kwiver::vital::bounding_box_d bbox( x1*bbox_out_width_rescale,
                                          y1*bbox_out_height_rescale,
                                          x2*bbox_out_width_rescale,
                                          y2*bbox_out_height_rescale );

      LOG_TRACE( m_logger, "Creating bounding box (" <<
                 std::to_string(bbox.min_x()) << ", " <<
                 std::to_string(bbox.max_x()) << ", " <<
                 std::to_string(bbox.min_y()) << ", " <<
                 std::to_string(bbox.max_y()) << ")" );

      auto dot = std::make_shared< detected_object_type >();
      dot->set_score( m_class_name, max_val );
      detected_objects->add( std::make_shared< kwiver::vital::detected_object >( bbox, max_val, dot ) );

      // Make CV rect for bbox so that we can remove it from consideration
      // during next iteration.
      cv::Rect cv_bbox( x1, y1, x2 - x1, y2 - y1 );
      heat_map(cv_bbox) = 0;

      ++cntr;
      if( cntr == m_max_boxes )
      {
        break;
      }
    }

    return detected_objects;
  }

  // --------------------------------------------------------------------------
  /// Threshold image and find connected components of binary image
  detected_object_set_sptr
  get_bbox_ccomponents( cv::Mat const &heat_map )
  {
    cv::Mat mask;
    cv::threshold( heat_map, mask, m_threshold, 1, cv::THRESH_BINARY );

    auto detected_objects = std::make_shared< detected_object_set >();

    std::vector< std::vector<cv::Point> > contours;

    // Remove outer border of pixels because findContours has trouble with
    // regions connected to the edge of the image.
    mask.row(0) = 0;
    mask.col(0) = 0;
    mask.row(mask.rows-1) = 0;
    mask.col(mask.cols-1) = 0;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0) );

    auto dot = std::make_shared< detected_object_type >();
    dot->set_score( m_class_name, m_fixed_score );

    for(unsigned j=0; j < contours.size(); ++j)
    {
      //LOG_DEBUG( logger(), "Contour " << j << ": " << std::to_string(contourArea(contours[j], false)));
      double area = cv::contourArea(contours[j]);
      if( area >= m_min_area && area <= m_max_area)
      {
        cv::Rect cv_bbox = cv::boundingRect(contours[j]);
        if( area >= cv_bbox.width*cv_bbox.height*m_min_fill_fraction )
        {
          kwiver::vital::bounding_box_d bbox( cv_bbox.x, cv_bbox.y,
                                              cv_bbox.x + cv_bbox.width,
                                              cv_bbox.y + cv_bbox.height );

          detected_objects->add(
            std::make_shared< kwiver::vital::detected_object >( bbox,
                                                                m_fixed_score,
                                                                dot ) );
        }
      }
    }
    LOG_TRACE( m_logger, "Finished creating bounding boxes");
    return detected_objects;
  }
  // --------------------------------------------------------------------------

};

/// Constructor
detect_heat_map
::detect_heat_map()
: d_(new priv)
{
  attach_logger( "arrows.ocv.detect_heat_map" );
  d_->m_logger = logger();
}

/// Destructor
detect_heat_map
::~detect_heat_map() noexcept
{
}

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_heat_map
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "threshold", d_->m_threshold,
                     "Threshold value applied to each pixel of the heat map to "
                     "turn it into a binary mask. Any pixels with value "
                     "strictly greater than this threshold will be turned on "
                     "in the mask. Detection objects will be associated with "
                     "connected-component regions of above-threshold pixels. "
                     "The default threshold of -1 indicates that further "
                     "processing will be done on the full-range heat map "
                     "image. This mode of processing requires that "
                     "'force_bbox_width' and 'force_bbox_height' be set." );
  config->set_value( "force_bbox_width", d_->m_force_bbox_width,
                     "Create bounding boxes of this fixed width." );
  config->set_value( "force_bbox_height", d_->m_force_bbox_height,
                     "Create bounding boxes of this fixed height." );
  config->set_value( "score_mode", d_->m_score_mode,
                     "Mode in which a score is attributed to each detected"
                     "object. A numerical value indicates that all detected"
                     "objects will be assigned this fixed score. No other"
                     "modes are defined at this time." );
  config->set_value( "bbox_buffer", d_->m_bbox_buffer,
                     "If a bounding box of fixed height and width is specified,"
                     "the default bbox_buffer of 0 indicates that the bounding"
                     "boxes will tightly crop features in the heat map, and "
                     "multiple, non-overlapping bounding boxes will be created "
                    "to cover large, extended heat-map features. With a value "
                     "greater than 0, generated bounding boxes will tend to "
                     "have that number of pixels of buffer from the heat-map "
                     "features. Also, setting bbox_buffer causes the generated "
                     "bounding boxes to tend to overlap by this number of "
                     "pixels when multiple boxes are required to cover and "
                     "extended heat-map feature." );
  config->set_value( "min_area", d_->m_min_area,
                     "Minimum area of above-threshold pixels in a connected "
                     "cluster allowed. Area is approximately equal to the "
                     "number of pixels in the cluster." );
  config->set_value( "max_area", d_->m_max_area,
                     "Maximum area of above-threshold pixels in a connected "
                     "cluster allowed. Area is approximately equal to the "
                     "number of pixels in the cluster." );
  config->set_value( "min_fill_fraction", d_->m_min_fill_fraction,
                     "Fraction of the bounding box filled with above threshold "
                     "pixels." );
  config->set_value( "class_name", d_->m_class_name, "Detection class name." );
  config->set_value( "max_boxes", d_->m_max_boxes, "Maximum number of "
                     "bounding boxes to generate. If exceeded, the top "
                     "'max_boxes' ones will be returned" );
  config->set_value( "pyr_red_levels", d_->m_pyr_red_levels, "Levels of image "
                     "pyramid reduction (decimation) on the heat map before "
                     "box selection. This improves speed at the expense of "
                     "coarseness of bounding box placement. " );

  return config;
}

/// Set this algo's properties via a config block
void
detect_heat_map
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, in_config );
  cd.warn_extra_keys( logger() );

  config->merge_config(in_config);

  d_->m_threshold          = config->get_value<int>( "threshold" );
  d_->m_force_bbox_width   = config->get_value<int>( "force_bbox_width" );
  d_->m_force_bbox_height  = config->get_value<int>( "force_bbox_height" );
  d_->m_bbox_buffer        = config->get_value<int>( "bbox_buffer" );
  d_->m_min_area           = config->get_value<int>( "min_area" );
  d_->m_max_area           = config->get_value<int>( "max_area" );
  d_->m_min_fill_fraction  = config->get_value<double>( "min_fill_fraction" );
  d_->m_class_name         = config->get_value<std::string>( "class_name" );
  d_->m_score_mode         = config->get_value<std::string>( "score_mode" );
  d_->m_max_boxes          = config->get_value<int>( "max_boxes" );
  d_->m_pyr_red_levels     = config->get_value<int>( "pyr_red_levels" );

  if( (d_->m_force_bbox_width == -1  && d_->m_force_bbox_height != -1) ||
      (d_->m_force_bbox_width != -1  && d_->m_force_bbox_height == -1) ||
      (d_->m_force_bbox_width != -1  && d_->m_force_bbox_width <= 0)   ||
      (d_->m_force_bbox_height != -1 && d_->m_force_bbox_height <= 0) )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "'force_bbox_width' and "
                                             "'force_bbox_height' must both be "
                                             "-1, indicating that a particular "
                                             "bounding box size will not be "
                                             "enforced, or both positive, "
                                             "indicating the size of the "
                                             "bounding box that will be "
                                             "enforced." );
  }

  if( d_->m_force_bbox_width > 0  && d_->m_force_bbox_height > 0 )
  {
    if( d_->m_force_bbox_width - d_->m_bbox_buffer <= 0 )
    {
      VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                               "(force_bbox_width - bbox_buffer) "
                                               "must be positive." );
    }

    if( d_->m_force_bbox_height - d_->m_bbox_buffer <= 0 )
    {
      VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                               "(force_bbox_height - "
                                               "bbox_buffer) must be "
                                               "positive." );
    }

    d_->m_force_bbox_size = true;
  }
  else if( d_->m_threshold == -1 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "If 'force_bbox_width' and "
                                             "'force_bbox_height' are not set,"
                                             "then a positive 'threshold' is "
                                             "required." );
  }

  if( d_->m_threshold < 0 && d_->m_threshold != -1 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "'threshold' must be non-negative "
                                             "in order for valid thresholding "
                                             "or equal to -1, indicating that "
                                             "no thresholding will be done." );
  }

  LOG_DEBUG( logger(), "threshold: " << std::to_string(d_->m_threshold) );
  LOG_DEBUG( logger(), "force_bbox_width: " << std::to_string(d_->m_force_bbox_width) );
  LOG_DEBUG( logger(), "force_bbox_height: " << std::to_string(d_->m_force_bbox_height) );
  LOG_DEBUG( logger(), "bbox_buffer: " << std::to_string(d_->m_bbox_buffer) );
  LOG_DEBUG( logger(), "min_area: " << std::to_string(d_->m_min_area) );
  LOG_DEBUG( logger(), "max_area: " << std::to_string(d_->m_max_area) );
  LOG_DEBUG( logger(), "min_fill_fraction: " << std::to_string(d_->m_min_fill_fraction) );
  LOG_DEBUG( logger(), "class_name: " << d_->m_class_name );

  // Extract a numerical score from the score_mode string if possible.
  char* p;
  double converted = strtod(d_->m_score_mode.c_str(), &p);
  if( !(*p) )
  {
    d_->m_score_mode = "fixed";
    d_->m_fixed_score = converted;
  }

  LOG_DEBUG( logger(), "score_mode: " << d_->m_score_mode );
  if( d_->m_score_mode == "fixed" )
  {
    LOG_DEBUG( logger(), "fixed_score: " << d_->m_fixed_score );
  }
}

bool
detect_heat_map
::check_configuration(vital::config_block_sptr config_in) const
{
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, config_in );
  return ! cd.warn_extra_keys( logger() );
}

/// Return homography to stabilize the image_src relative to the key frame
detected_object_set_sptr
detect_heat_map
::detect(image_container_sptr image_data) const
{
  if ( !image_data)
  {
    VITAL_THROW( vital::invalid_data, "Inputs to ocv::detect_heat_map are null");
  }
  LOG_TRACE( logger(), "Received image");

  const cv::Mat cv_src = ocv::image_container::vital_to_ocv(image_data->get_image(), ocv::image_container::BGR_COLOR );

  if ( cv_src.channels() > 1 )
  {
    VITAL_THROW( vital::invalid_data, "Heat map image must be single channel.");
  }

  return d_->get_bounding_boxes(cv_src);
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
