/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "write_detected_object_set_viame_csv.h"

#include "convert_notes_to_attributes.h"

#include <vital/util/tokenize.h>

#include <memory>
#include <vector>
#include <fstream>
#include <ctime>

#if ( __GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__) )
  #include <cstdatomic>
#else
  #include <atomic>
#endif

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>

  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>

  #include <queue>

  static std::vector< cv::Point >
  simplify_polygon( std::vector< cv::Point > const& curve, size_t max_points );
#endif

namespace viame {


// --------------------------------------------------------------------------------
void
write_detected_object_set_viame_csv
::initialize()
{
  m_first = true;
  m_frame_number = 0;
}


// --------------------------------------------------------------------------------
bool
write_detected_object_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// --------------------------------------------------------------------------------
void
write_detected_object_set_viame_csv
::write_set( const kwiver::vital::detected_object_set_sptr set,
           std::string const& image_name )
{
  if( c_mask_to_poly_tol >= 0 && c_mask_to_poly_points >= 0 )
  {
    throw std::runtime_error(
      "At most one of use mask_to_poly_tol and mask_to_poly_points "
      "can be enabled (nonnegative)" );
  }
#ifndef VIAME_ENABLE_OPENCV
  if( c_mask_to_poly_tol >= 0 || c_mask_to_poly_points >= 0 )
  {
    throw std::runtime_error(
      "Must have OpenCV enabled to use mask_to_poly_tol or mask_to_poly_points" );
  }
#endif

  if( m_first )
  {
    std::time_t current_time;
    struct tm* timeinfo;

    time( &current_time );
    timeinfo = localtime ( &current_time );
    char* cp = asctime( timeinfo );
    cp[ strlen( cp )-1 ] = 0; // remove trailing newline
    const std::string formatted_time( cp );

    // Write file header(s)
    stream() << "# 1: Detection or Track-id,"
             << "  2: Video or Image Identifier,"
             << "  3: Unique Frame Identifier,"
             << "  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),"
             << "  8: Detection or Length Confidence,"
             << "  9: Target Length (0 or -1 if invalid),"
             << "  10-11+: Repeated Species, Confidence Pairs or Attributes"
             << std::endl;

    // Write metadata(s)
    stream() << "# metadata";

    if( !c_frame_rate.empty() )
    {
      stream() << ", fps: " << c_frame_rate;
    }

    stream() << ", exported_by: write_detected_object_set_viame_csv";
    stream() << ", exported_at: " << formatted_time;

    if( !c_model_identifier.empty() )
    {
      stream() << ", model: " << c_model_identifier;
    }
    if( !c_version_identifier.empty() )
    {
      stream() << ", software: " << c_version_identifier;
    }
    stream() << std::endl;

    m_first = false;
  } // end first

  // process all detections if a valid input set was provided
  if( !set )
  {
    ++m_frame_number;
    return;
  }

  auto ie = set->cend();

  for( auto det = set->cbegin(); det != ie; ++det )
  {
    const kwiver::vital::bounding_box_d bbox( (*det)->bounding_box() );

    static std::atomic<unsigned> id_counter( 0 );
    const unsigned det_id = id_counter++;

    std::string video_id;
    if( !c_stream_identifier.empty() )
    {
      video_id = c_stream_identifier;
    }
    else
    {
      video_id = image_name;
      const size_t last_slash_idx = video_id.find_last_of("\\/");
      if ( std::string::npos != last_slash_idx )
      {
        video_id.erase( 0, last_slash_idx + 1 );
      }
    }

    stream() << det_id << ","               // 1: track id
             << video_id << ",";            // 2: video or image id

    if( c_write_frame_number )
    {
      stream() << m_frame_number << ","; // 3: frame number
    }
    else
    {
      stream() << image_name << ",";        // 3: frame identfier
    }

    stream() << bbox.min_x() << ","         // 4: TL-x
             << bbox.min_y() << ","         // 5: TL-y
             << bbox.max_x() << ","         // 6: BR-x
             << bbox.max_y() << ","         // 7: BR-y
             << (*det)->confidence() << ",";// 8: confidence

    // 9: length - read from attributes if available
    if( (*det)->has_attribute( "length" ) )
    {
      try
      {
        stream() << (*det)->get_attribute< double >( "length" );
      }
      catch( ... )
      {
        stream() << "0";
      }
    }
    else
    {
      stream() << "0";
    }

    const auto dot = (*det)->type();

    if( dot )
    {
      for( auto name : dot->top_class_names( c_top_n_classes ) )
      {
        // Write out the <name> <score> pair
        stream() << "," << name << "," << dot->score( name );
      }
    }

    // Preferentially write out the explicit polygon
    if( !(*det)->polygon().empty() &&
        ( ( c_mask_to_poly_tol < 0 &&
            c_mask_to_poly_points < 0 ) ||
           !(*det)->mask() ) )
    {
      stream() << ",(poly)";
      auto poly = (*det)->polygon();
      for( auto&& p : poly )
      {
        stream() << " " << p[0] << " " << p[1];
      }
    }
#ifdef VIAME_ENABLE_OPENCV
    else if( (*det)->mask() && ( c_mask_to_poly_tol >= 0 ||
                                 c_mask_to_poly_points >= 0 ) )
    {
      using ic = kwiver::arrows::ocv::image_container;
      auto ref_x = static_cast< int >( bbox.min_x() );
      auto ref_y = static_cast< int >( bbox.min_y() );
      cv::Mat mask = ic::vital_to_ocv( (*det)->mask()->get_image(),
                                       ic::OTHER_COLOR );
      std::vector< std::vector< cv::Point > > contours;
      std::vector< cv::Vec4i > hierarchy;
      // Pre-3.2 OpenCV may modify the passed image, so we clone it.
      cv::findContours( mask.clone(), contours, hierarchy,
                        cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
      for( size_t i = 0; i < contours.size(); ++i )
      {
        auto& contour = contours[i];
        int x_min, x_max, y_min, y_max;
        x_min = x_max = contour[0].x;
        y_min = y_max = contour[0].y;
        for( size_t j = 1; j < contour.size(); ++j )
        {
          x_min = std::min( x_min, contour[j].x );
          x_max = std::max( x_max, contour[j].x );
          y_min = std::min( y_min, contour[j].y );
          y_max = std::max( y_max, contour[j].y );
        }
        std::vector< cv::Point > simp_contour;
        if( c_mask_to_poly_tol >= 0 )
        {
          double tol = c_mask_to_poly_tol * std::min( x_max - x_min + 1,
                                                       y_max - y_min + 1 );
          cv::approxPolyDP( contour, simp_contour, tol, /*closed:*/ true );
        }
        else
        {
          simp_contour = simplify_polygon( contour, c_mask_to_poly_points );
        }
        stream() << ( hierarchy[i][3] < 0 ? ",(poly)" : ",(hole)" );
        for( auto&& p : simp_contour )
        {
          stream() << " " << p.x + ref_x << " " << p.y + ref_y;
        }
      }
    }
#endif

    if( !(*det)->keypoints().empty() )
    {
      for( const auto& kp : (*det)->keypoints() )
      {
        stream() << "," << "(kp) " << kp.first;
        stream() << " " << kp.second.value()[0] << " " << kp.second.value()[1];
      }
    }

    if( !(*det)->notes().empty() )
    {
      stream() << notes_to_attributes( (*det)->notes(), "," );
    }

    stream() << std::endl;
  }

  // Flush stream to prevent buffer issues
  stream().flush();

  // Put each set on a new frame
  ++m_frame_number;
}

} // end namespace

#ifdef VIAME_ENABLE_OPENCV
static std::vector< cv::Point >
simplify_polygon( std::vector< cv::Point > const& curve, size_t max_points )
{
  // Modified Ramer-Douglas-Peucker.  Instead of keeping points out of
  // tolerance, we add points until we reach the max.
  size_t size = curve.size();
  max_points = std::max( max_points, size_t( 2 ) );
  if( size <= max_points )
  {
    return curve;
  }

  // Find approximate diameter endpoints
  size_t start = 0, opposite;
  for( int diameter_iter = 0; diameter_iter < 3; ++diameter_iter )
  {
    auto& ps = curve[ start ];
    size_t i_max = start; int sq_dist_max = 0;
    for( size_t i = 0; i < size; ++i ) {
      int dx = curve[ i ].x - ps.x, dy = curve[ i ].y - ps.y;
      int sq_dist = dx * dx + dy * dy;
      if( sq_dist > sq_dist_max )
      {
        i_max = i;
        sq_dist_max = sq_dist;
      }
    }
    opposite = start;
    start = i_max;
  }

  // Indices for rec and find_max are relative to start
  auto to_rel = [&]( size_t i ) { return ( i + size - start ) % size; };
  auto from_rel = [&]( size_t i ) { return ( i + start ) % size; };

  struct rec
  {
    double sq_dist; size_t l, r, i;
    bool operator <( rec const& other ) const
    {
      return this->sq_dist < other.sq_dist;
    }
  };

  auto find_max = [&]( size_t l, size_t r )
  {
    auto& pl = curve[ from_rel( l ) ]; auto& pr = curve[ from_rel( r ) ];
    double dx = pr.x - pl.x, dy = pr.y - pl.y;
    double sq_dist_den = dx * dx + dy * dy;

    auto sqrt_sq_dist_num = [&]( size_t i )
    {
      auto& p = curve[ from_rel( i ) ];
      return std::abs( ( p.x - pl.x ) * dy - ( p.y - pl.y ) * dx );
    };

    size_t i = l + 1;
    size_t i_max = i; double ssdn_max = sqrt_sq_dist_num( i );

    for( ++i; i < r; ++i )
    {
      auto ssdn = sqrt_sq_dist_num( i );
      if( ssdn > ssdn_max )
      {
        i_max = i;
        ssdn_max = ssdn;
      }
    }
    return rec{ ssdn_max * ssdn_max / sq_dist_den, l, r, i_max };
  };

  // Initialize using the two approximate diameter endpoints and the
  // parts of the curve between them.
  std::vector< bool > keep( size, false );
  keep[ start ] = keep[ opposite ] = true;
  std::priority_queue< rec > queue;
  queue.push( find_max( 0, to_rel( opposite ) ) );
  queue.push( find_max( to_rel( opposite ), size ) );

  for( size_t keep_count = 2; keep_count < max_points; ++keep_count )
  {
    auto max_rec = queue.top(); queue.pop();
    keep[ from_rel( max_rec.i ) ] = true;
    if( max_rec.i - max_rec.l > 1 )
    {
      queue.push( find_max( max_rec.l, max_rec.i ) );
    }
    if( max_rec.r - max_rec.i > 1 )
    {
      queue.push( find_max( max_rec.i, max_rec.r ) );
    }
  }

  std::vector< cv::Point > result;
  for( size_t i = 0; i < size; ++i )
  {
    if( keep[ i ] )
    {
      result.push_back( curve[ i ] );
    }
  }
  return result;
}
#endif
