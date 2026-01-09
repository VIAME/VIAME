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
class write_detected_object_set_viame_csv::priv
{
public:
  priv( write_detected_object_set_viame_csv* parent)
    : m_parent( parent )
    , m_first( true )
    , m_frame_number( 0 )
    , m_write_frame_number( true )
    , m_stream_identifier( "" )
    , m_model_identifier( "" )
    , m_version_identifier( "" )
    , m_frame_rate( "" )
    , m_mask_to_poly_tol( -1 )
    , m_mask_to_poly_points( 20 )
    , m_top_n_classes( 0 )
  {}

  ~priv() {}

  write_detected_object_set_viame_csv* m_parent;
  bool m_first;
  int m_frame_number;
  bool m_write_frame_number;
  std::string m_stream_identifier;
  std::string m_model_identifier;
  std::string m_version_identifier;
  std::string m_frame_rate;
  double m_mask_to_poly_tol;
  int m_mask_to_poly_points;
  unsigned m_top_n_classes;
};


// ================================================================================
write_detected_object_set_viame_csv
::write_detected_object_set_viame_csv()
  : d( new write_detected_object_set_viame_csv::priv( this ) )
{
  attach_logger( "viame.core.write_detected_object_set_viame_csv" );
}


write_detected_object_set_viame_csv
::~write_detected_object_set_viame_csv()
{
}


// --------------------------------------------------------------------------------
void
write_detected_object_set_viame_csv
::set_configuration( kwiver::vital::config_block_sptr config_in )
{
  kwiver::vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  d->m_write_frame_number =
    config->get_value< bool >( "write_frame_number" );
  d->m_stream_identifier =
    config->get_value< std::string >( "stream_identifier" );
  d->m_model_identifier =
    config->get_value< std::string >( "model_identifier" );
  d->m_version_identifier =
    config->get_value< std::string >( "version_identifier" );
  d->m_frame_rate =
    config->get_value< std::string >( "frame_rate" );
  d->m_mask_to_poly_tol =
    config->get_value< double >( "mask_to_poly_tol" );
  d->m_mask_to_poly_points =
    config->get_value< int >( "mask_to_poly_points" );
  d->m_top_n_classes =
    config->get_value< unsigned >( "top_n_classes" );

  if( d->m_mask_to_poly_tol >= 0 && d->m_mask_to_poly_points >= 0 )
  {
    throw std::runtime_error(
      "At most one of use mask_to_poly_tol and mask_to_poly_points "
      "can be enabled (nonnegative)" );
  }
#ifndef VIAME_ENABLE_OPENCV
  if( d->m_mask_to_poly_tol >= 0 || d->m_mask_to_poly_points >= 0 )
  {
    throw std::runtime_error(
      "Must have OpenCV enabled to use mask_to_poly_tol or mask_to_poly_points" );
  }
#endif
}


// --------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
write_detected_object_set_viame_csv
::get_configuration() const
{
  // get base config from base class
  kwiver::vital::config_block_sptr config = algorithm::get_configuration();

  // Class parameters
  config->set_value( "write_frame_number", d->m_write_frame_number,
    "Write a frame number for the unique frame ID field (as opposed to a string "
    "identifier) for column 3 in the output csv." );
  config->set_value( "stream_identifier", d->m_stream_identifier,
    "Optional fixed video name over-ride to write to output column 2 in the csv." );
  config->set_value( "model_identifier", d->m_model_identifier,
    "Model identifier string to write to the header or the csv." );
  config->set_value( "frame_rate", d->m_frame_rate,
    "Frame rate string to write to the header or the csv." );
  config->set_value( "version_identifier", d->m_version_identifier,
    "Version identifier string to write to the header or the csv." );
  config->set_value( "mask_to_poly_tol", d->m_mask_to_poly_tol,
    "Write segmentation masks when available as polygons with the specified "
    "relative tolerance for the conversion.  Set to a negative value to disable." );
  config->set_value( "mask_to_poly_points", d->m_mask_to_poly_points,
    "Write segmentation masks when available as polygons with the specified "
    "maximum number of points.  Set to a negative value to disable." );
  config->set_value( "top_n_classes", d->m_top_n_classes,
    "Only print out this maximum number of classes (highest score first)" );

  return config;
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
  if( d->m_first )
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

    if( !d->m_frame_rate.empty() )
    {
      stream() << ", fps: " << d->m_frame_rate;
    }

    stream() << ", exported_by: write_detected_object_set_viame_csv";
    stream() << ", exported_at: " << formatted_time;

    if( !d->m_model_identifier.empty() )
    {
      stream() << ", model: " << d->m_model_identifier;
    }
    if( !d->m_version_identifier.empty() )
    {
      stream() << ", software: " << d->m_version_identifier;
    }
    stream() << std::endl;

    d->m_first = false;
  } // end first

  // process all detections if a valid input set was provided
  if( !set )
  {
    ++d->m_frame_number;
    return;
  }

  auto ie = set->cend();

  for( auto det = set->cbegin(); det != ie; ++det )
  {
    const kwiver::vital::bounding_box_d bbox( (*det)->bounding_box() );

    static std::atomic<unsigned> id_counter( 0 );
    const unsigned det_id = id_counter++;

    std::string video_id;
    if( !d->m_stream_identifier.empty() )
    {
      video_id = d->m_stream_identifier;
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

    if( d->m_write_frame_number )
    {
      stream() << d->m_frame_number << ","; // 3: frame number
    }
    else
    {
      stream() << image_name << ",";        // 3: frame identfier
    }

    stream() << bbox.min_x() << ","         // 4: TL-x
             << bbox.min_y() << ","         // 5: TL-y
             << bbox.max_x() << ","         // 6: BR-x
             << bbox.max_y() << ","         // 7: BR-y
             << (*det)->confidence() << "," // 8: confidence
             << "0";                        // 9: length

    const auto dot = (*det)->type();

    if( dot )
    {
      for( auto name : dot->top_class_names( d->m_top_n_classes ) )
      {
        // Write out the <name> <score> pair
        stream() << "," << name << "," << dot->score( name );
      }
    }

    // Preferentially write out the explicit polygon
    if( !(*det)->polygon().empty() &&
        ( ( d->m_mask_to_poly_tol < 0 &&
            d->m_mask_to_poly_points < 0 ) ||
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
    else if( (*det)->mask() && ( d->m_mask_to_poly_tol >= 0 ||
                                 d->m_mask_to_poly_points >= 0 ) )
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
        if( d->m_mask_to_poly_tol >= 0 )
        {
          double tol = d->m_mask_to_poly_tol * std::min( x_max - x_min + 1,
                                                         y_max - y_min + 1 );
          cv::approxPolyDP( contour, simp_contour, tol, /*closed:*/ true );
        }
        else
        {
          simp_contour = simplify_polygon( contour, d->m_mask_to_poly_points );
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
  ++d->m_frame_number;
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
