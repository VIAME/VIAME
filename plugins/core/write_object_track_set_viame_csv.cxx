/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of detected object set csv output
 */

#include "write_object_track_set_viame_csv.h"

#include "convert_notes_to_attributes.h"
#include "utilities_segmentation.h"

#include <ctime>
#include <sstream>
#include <iomanip>

#ifdef VIAME_ENABLE_OPENCV
#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

namespace viame {

namespace kv = kwiver::vital;

// Helper function for computing average track object type
static kv::detected_object_type_sptr
compute_average_tot( kv::track_sptr trk_ptr,
                     bool weighted = false,
                     bool scale_by_conf = false,
                     std::string ignore_class = "" )
{
  if( !trk_ptr )
  {
    return kv::detected_object_type_sptr();
  }

  std::vector< std::string > output_names;
  std::vector< double > output_scores;

  double weighted_mass = 0.0;
  double weighted_non_ignore_mass = 0.0;
  double weighted_ignore_mass = 0.0;

  std::map< std::string, double > class_sum;
  double ignore_sum = 0.0;
  double conf_sum = 0.0;
  unsigned conf_count = 0;

  for( auto ts_ptr : *trk_ptr )
  {
    kv::object_track_state* ts =
      static_cast< kv::object_track_state* >( ts_ptr.get() );

    if( !ts->detection() )
    {
      continue;
    }

    kv::detected_object_type_sptr dot = ts->detection()->type();

    if( dot )
    {
      double weight = ( weighted ? ts->detection()->confidence() : 1.0 );

      if( scale_by_conf )
      {
        conf_sum += ts->detection()->confidence();
        conf_count += 1;
      }

      bool ignore = ( dot->class_names().size() == 1 &&
                      dot->class_names()[0] == ignore_class );

      if( ignore )
      {
        ignore_sum += ( dot->score( ignore_class ) * weight );
        weighted_ignore_mass += weight;
      }
      else
      {
        for( const auto& name : dot->class_names() )
        {
          class_sum[ name ] += ( dot->score( name ) * weight );
        }
        weighted_non_ignore_mass += weight;
      }

      weighted_mass += weight;
    }
  }

  double prob_scale_factor = 1.0;

  if( scale_by_conf && conf_count > 0 )
  {
    prob_scale_factor = 0.1 + 0.9 * ( conf_sum / conf_count );
  }

  if( weighted_mass > 0.0 && weighted_ignore_mass == 0.0 )
  {
    prob_scale_factor /= weighted_mass;
  }
  else if( weighted_ignore_mass > 0.0 && weighted_non_ignore_mass > 0.0 )
  {
    prob_scale_factor /= weighted_non_ignore_mass;
  }
  else if( weighted_ignore_mass > 0.0 )
  {
    class_sum[ ignore_class ] = ignore_sum;
    prob_scale_factor /= weighted_ignore_mass;
  }

  for( auto itr : class_sum )
  {
    output_names.push_back( itr.first );
    output_scores.push_back( prob_scale_factor * itr.second );
  }

  if( output_names.empty() )
  {
    return kv::detected_object_type_sptr();
  }

  return std::make_shared< kv::detected_object_type >(
    output_names, output_scores );
}


// ===============================================================================
void
write_object_track_set_viame_csv
::initialize()
{
  m_logger = kv::get_logger( "write_object_track_set_viame_csv" );
  m_first = true;
  m_start_time = 0;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::set_configuration_internal( kv::config_block_sptr config )
{
  if( !c_active_writing )
  {
    time( &m_start_time );
  }
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
}


// -------------------------------------------------------------------------------
std::string
write_object_track_set_viame_csv
::format_image_id( const kv::object_track_state* ts )
{
  if( c_write_time_as_uid )
  {
    char output[10];
    const kv::time_usec_t usec( 1000000 );
    const kv::time_usec_t time_s = ts->time() / usec;
    unsigned time_us = ts->time() % usec;
    std::string time_us_str = std::to_string( time_us );
    while( time_us_str.size() < 6 )
    {
      time_us_str = "0" + time_us_str;
    }
    struct tm* tmp = gmtime( &time_s );
    strftime( output, sizeof( output ), "%H:%M:%S", tmp );
    return std::string( output ) + "." + time_us_str;
  }
  else if( !m_frame_uids.empty() )
  {
    std::string fileuid = m_frame_uids[ static_cast<unsigned>( ts->frame() ) ];

    const size_t last_slash_idx = fileuid.find_last_of("\\/");

    if( std::string::npos != last_slash_idx )
    {
      fileuid.erase( 0, last_slash_idx + 1 );
    }

    return fileuid;
  }
  else
  {
    return c_stream_identifier;
  }
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::write_header_info( std::ostream &stream )
{
  std::time_t current_time;
  struct tm* timeinfo;

  time( &current_time );
  timeinfo = localtime( &current_time );
  char* cp = asctime( timeinfo );
  cp[ strlen( cp )-1 ] = 0; // remove trailing newline
  const std::string formatted_time( cp );

  // Write file header(s)
  stream << "# 1: Detection or Track-id,"
         << "  2: Video or Image Identifier,"
         << "  3: Unique Frame Identifier,"
         << "  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),"
         << "  8: Detection or Length Confidence,"
         << "  9: Target Length (0 or -1 if invalid),"
         << "  10-11+: Repeated Species, Confidence Pairs or Attributes"
         << std::endl;

  stream << "# metadata";

  if( !c_frame_rate.empty() )
  {
    stream << ", fps: " << c_frame_rate;
  }
  if( m_start_time )
  {
    stream << ", exec_time: " << std::difftime( current_time, m_start_time );
  }

  stream << ", exported_by: write_object_track_set_viame_csv";
  stream << ", exported_at: " << formatted_time;

  if( !c_model_identifier.empty() )
  {
    stream << ", model: " << c_model_identifier;
  }
  if( !c_version_identifier.empty() )
  {
    stream << ", software: " << c_version_identifier;
  }
  stream << std::endl;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::write_detection_info( std::ostream &stream,
                        const kv::detected_object_sptr &det )
{
  // Sanity return in case method was called with empty detection
  if( !det )
    return;

  auto bbox = det->bounding_box();

  // Preferentially write out the explicit polygon
  if( !det->polygon().empty() &&
      ( ( c_mask_to_poly_tol < 0 &&
          c_mask_to_poly_points < 0 ) ||
        !det->mask() ) )
  {
    stream << c_delimiter << "(poly)";
    auto poly = det->polygon();
    for( auto&& p : poly )
    {
      stream << " " << p[0] << " " << p[1];
    }
  }
#ifdef VIAME_ENABLE_OPENCV
  else if( det->mask() && ( c_mask_to_poly_tol >= 0 ||
                            c_mask_to_poly_points >= 0 ) )
  {
    using ic = kwiver::arrows::ocv::image_container;
    auto ref_x = static_cast< int >( bbox.min_x() );
    auto ref_y = static_cast< int >( bbox.min_y() );
    cv::Mat mask = ic::vital_to_ocv( det->mask()->get_image(),
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
      std::vector< kwiver::vital::point_2i > simp_contour;
      if( c_mask_to_poly_tol >= 0 )
      {
        double tol = c_mask_to_poly_tol * std::min( x_max - x_min + 1,
                                                    y_max - y_min + 1 );
        std::vector< cv::Point > approx;
        cv::approxPolyDP( contour, approx, tol, /*closed:*/ true );
        for( auto const& p : approx )
        {
          simp_contour.emplace_back( p.x, p.y );
        }
      }
      else
      {
        std::vector< kwiver::vital::point_2i > kwiver_contour;
        kwiver_contour.reserve( contour.size() );
        for( auto const& p : contour )
        {
          kwiver_contour.emplace_back( p.x, p.y );
        }
        simp_contour = simplify_polygon( kwiver_contour, c_mask_to_poly_points );
      }
      stream << ( hierarchy[i][3] < 0 ? c_delimiter  + "(poly)" : c_delimiter  + "(hole)" );
      for( auto const& p : simp_contour )
      {
        stream << " " << p[ 0 ] + ref_x << " " << p[ 1 ] + ref_y;
      }
    }
  }
#endif

  if( !det->keypoints().empty() )
  {
    for( const auto& kp : det->keypoints() )
    {
      stream << c_delimiter << "(kp) " << kp.first;
      stream << " " << kp.second.value()[0] << " " << kp.second.value()[1];
    }
  }

  if( !det->notes().empty() )
  {
    stream << notes_to_attributes( det->notes(), c_delimiter );
  }
}


// -------------------------------------------------------------------------------
void write_object_track_set_viame_csv
::close()
{
  if( c_active_writing )
  {
    // No flushing required
    write_object_track_set::close();
    return;
  }

  write_header_info( stream() );

  for( auto trk_pair : m_tracks )
  {
    auto trk_ptr = trk_pair.second;

    const kv::detected_object_type_sptr trk_average_tot =
          ( c_tot_option == "detection" ? kv::detected_object_type_sptr()
            : compute_average_tot( trk_ptr,
                c_tot_option.find( "weighted" ) != std::string::npos,
                c_tot_option.find( "scaled_by_conf" ) != std::string::npos,
                c_tot_ignore_class ) );

    for( auto ts_ptr : *trk_ptr )
    {
      kv::object_track_state* ts =
        dynamic_cast< kv::object_track_state* >( ts_ptr.get() );

      if( !ts )
      {
        LOG_ERROR( m_logger,
          "Invalid timestamp " << trk_ptr->id() << " " << trk_ptr->size() );
        continue;
      }

      kv::detected_object_sptr det = ts->detection();
      const kv::bounding_box_d empty_box =
        kv::bounding_box_d( -1, -1, -1, -1 );
      kv::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );
      auto confidence = ( det ? det->confidence() : 0 );
      kv::frame_id_t frame_id = ts->frame() + c_frame_id_adjustment;

      stream() << trk_ptr->id() << c_delimiter            // 1: track id
               << format_image_id( ts ) << c_delimiter    // 2: video or image id
               << frame_id << c_delimiter                 // 3: frame number
               << bbox.min_x() << c_delimiter             // 4: TL-x
               << bbox.min_y() << c_delimiter             // 5: TL-y
               << bbox.max_x() << c_delimiter             // 6: BR-x
               << bbox.max_y() << c_delimiter             // 7: BR-y
               << confidence << c_delimiter;              // 8: confidence

      // 9: length - read from track attributes if available, fallback to detection
      double length_value = 0.0;
      bool found_length = false;
      if( trk_ptr->has_attribute( "length" ) )
      {
        try
        {
          length_value = trk_ptr->get_attribute< double >( "length" );
          found_length = true;
        }
        catch( ... ) {}
      }
      if( !found_length && det && det->has_attribute( "length" ) )
      {
        try
        {
          length_value = det->get_attribute< double >( "length" );
          found_length = true;
        }
        catch( ... ) {}
      }
      stream() << length_value;

      if( det )
      {
        const kv::detected_object_type_sptr dot =
          ( c_tot_option == "detection" ? det->type() : trk_average_tot );

        if( dot )
        {
          for( auto name : dot->top_class_names( c_top_n_classes ) )
          {
            stream() << c_delimiter << name << c_delimiter << dot->score( name );
          }
        }

        write_detection_info( stream(), det );

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }

  write_object_track_set::close();
}


// -------------------------------------------------------------------------------
bool
write_object_track_set_viame_csv
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::write_set( const kv::object_track_set_sptr& set,
             const kv::timestamp& ts,
             const std::string& file_id )
{
  if( m_first )
  {
    if( c_active_writing )
    {
      write_header_info( stream() );
    }

    m_first = false;
  }

  if( !file_id.empty() && ts.has_valid_frame() )
  {
    m_frame_uids[ static_cast<unsigned>( ts.get_frame() ) ] = file_id;
  }

  if( !set )
  {
    return;
  }

  if( !c_active_writing )
  {
    for( auto trk : set->tracks() )
    {
      m_tracks[ static_cast<unsigned>( trk->id() ) ] = trk;
    }
  }
  else
  {
    for( auto trk_ptr : set->tracks() )
    {
      if( !trk_ptr || trk_ptr->empty() )
      {
        LOG_ERROR( m_logger, "Received invalid track" );
        continue;
      }

      kv::object_track_state* state =
        dynamic_cast< kv::object_track_state* >( trk_ptr->back().get() );

      if( !state )
      {
        LOG_ERROR( m_logger, "Invalid track state for track "
                              << trk_ptr->id()
                              << " of length "
                              << trk_ptr->size() );
        continue;
      }

      if( state->frame() != ts.get_frame() )
      {
        // Last state is in the past, it was already written.
        continue;
      }

      kv::detected_object_sptr det = state->detection();

      const kv::bounding_box_d empty_box =
        kv::bounding_box_d( -1, -1, -1, -1 );

      kv::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );

      auto confidence = ( det ? det->confidence() : 0 );
      kv::frame_id_t frame_id = state->frame() + c_frame_id_adjustment;

      stream() << trk_ptr->id() << c_delimiter               // 1: track id
               << format_image_id( state ) << c_delimiter    // 2: video or image id
               << frame_id << c_delimiter                    // 3: frame number
               << bbox.min_x() << c_delimiter                // 4: TL-x
               << bbox.min_y() << c_delimiter                // 5: TL-y
               << bbox.max_x() << c_delimiter                // 6: BR-x
               << bbox.max_y() << c_delimiter                // 7: BR-y
               << confidence << c_delimiter;                 // 8: confidence

      // 9: length - read from track attributes if available, fallback to detection
      double length_value = 0.0;
      bool found_length = false;
      if( trk_ptr->has_attribute( "length" ) )
      {
        try
        {
          length_value = trk_ptr->get_attribute< double >( "length" );
          found_length = true;
        }
        catch( ... ) {}
      }
      if( !found_length && det && det->has_attribute( "length" ) )
      {
        try
        {
          length_value = det->get_attribute< double >( "length" );
          found_length = true;
        }
        catch( ... ) {}
      }
      stream() << length_value;

      if( det )
      {
        const kv::detected_object_type_sptr dot =
          ( c_tot_option == "detection" ? det->type() :
            compute_average_tot( trk_ptr,
              c_tot_option.find( "weighted" ) != std::string::npos,
              c_tot_option.find( "scaled_by_conf" ) != std::string::npos,
              c_tot_ignore_class ) );

        if( dot )
        {
          for( auto name : dot->top_class_names( c_top_n_classes ) )
          {
            stream() << c_delimiter << name << c_delimiter << dot->score( name );
          }
        }

        write_detection_info( stream(), det );

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }
}

} // end namespace viame
