/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of detected object set csv output
 */

#include "write_object_track_set_viame_csv.h"

#include "utilities_target_clfr.h"
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


// -------------------------------------------------------------------------------
class write_object_track_set_viame_csv::priv
{
public:
  priv( write_object_track_set_viame_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "write_object_track_set_viame_csv" ) )
    , m_first( true )
    , m_delim( "," )
    , m_stream_identifier( "" )
    , m_model_identifier( "" )
    , m_version_identifier( "" )
    , m_frame_rate( "" )
    , m_active_writing( false )
    , m_write_time_as_uid( false )
    , m_tot_option( "weighted_average" )
    , m_tot_ignore_class( "" )
    , m_frame_id_adjustment( 0 )
    , m_top_n_classes( 0 )
    , m_mask_to_poly_tol( -1 )
    , m_mask_to_poly_points( 20 )
    , m_start_time( 0 )
  { }

  ~priv() { }

  write_object_track_set_viame_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  std::string m_delim;
  std::string m_stream_identifier;
  std::string m_model_identifier;
  std::string m_version_identifier;
  std::string m_frame_rate;
  std::map< unsigned, kwiver::vital::track_sptr > m_tracks;
  bool m_active_writing;
  bool m_write_time_as_uid;

  std::string m_tot_option;
  std::string m_tot_ignore_class;
  int m_frame_id_adjustment;
  std::map< unsigned, std::string > m_frame_uids;
  unsigned m_top_n_classes;
  double m_mask_to_poly_tol;
  int m_mask_to_poly_points;
  std::time_t m_start_time;

  std::string format_image_id( const kwiver::vital::object_track_state* ts );
  void write_header_info( std::ostream& stream );
  void write_detection_info( std::ostream& stream,
                             const kwiver::vital::detected_object_sptr& det );
};

std::string
write_object_track_set_viame_csv::priv
::format_image_id( const kwiver::vital::object_track_state* ts )
{
  if( m_write_time_as_uid )
  {
    char output[10];
    const kwiver::vital::time_usec_t usec( 1000000 );
    const kwiver::vital::time_usec_t time_s = ts->time() / usec;
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
    return m_stream_identifier;
  }
}

void write_object_track_set_viame_csv::priv::write_header_info(
  std::ostream &stream )
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

  if( !m_frame_rate.empty() )
  {
    stream << ", fps: " << m_frame_rate;
  }
  if( m_start_time )
  {
    stream << ", exec_time: " << std::difftime( current_time, m_start_time );
  }

  stream << ", exported_by: write_object_track_set_viame_csv";
  stream << ", exported_at: " << formatted_time;

  if( !m_model_identifier.empty() )
  {
    stream << ", model: " << m_model_identifier;
  }
  if( !m_version_identifier.empty() )
  {
    stream << ", software: " << m_version_identifier;
  }
  stream << std::endl;
}

void write_object_track_set_viame_csv::priv::write_detection_info(
  std::ostream &stream,
  const kwiver::vital::detected_object_sptr &det )
{
  // Sanity return in case method was called with empty detection
  if( !det )
    return;

  auto bbox = det->bounding_box();

  // Preferentially write out the explicit polygon
  if( !det->polygon().empty() &&
      ( ( m_mask_to_poly_tol < 0 &&
          m_mask_to_poly_points < 0 ) ||
        !det->mask() ) )
  {
    stream << m_delim << "(poly)";
    auto poly = det->polygon();
    for( auto&& p : poly )
    {
      stream << " " << p[0] << " " << p[1];
    }
  }
#ifdef VIAME_ENABLE_OPENCV
  else if( det->mask() && ( m_mask_to_poly_tol >= 0 ||
                            m_mask_to_poly_points >= 0 ) )
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
      if( m_mask_to_poly_tol >= 0 )
      {
        double tol = m_mask_to_poly_tol * std::min( x_max - x_min + 1,
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
        simp_contour = simplify_polygon( kwiver_contour, m_mask_to_poly_points );
      }
      stream << ( hierarchy[i][3] < 0 ? m_delim  + "(poly)" : m_delim  + "(hole)" );
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
      stream << m_delim << "(kp) " << kp.first;
      stream << " " << kp.second.value()[0] << " " << kp.second.value()[1];
    }
  }

  if( !det->notes().empty() )
  {
    stream << notes_to_attributes( det->notes(), m_delim );
  }
}

kwiver::vital::detected_object_type_sptr
compute_average_tot( kwiver::vital::track_sptr trk_ptr,
                     bool weighted = false,
                     bool scale_by_conf = false,
                     std::string ignore_class = "" )
{
  if( !trk_ptr )
  {
    return kwiver::vital::detected_object_type_sptr();
  }

  // Extract detections from track states
  std::vector< kwiver::vital::detected_object_sptr > detections;
  for( auto ts_ptr : *trk_ptr )
  {
    auto* ts = static_cast< kwiver::vital::object_track_state* >( ts_ptr.get() );
    if( ts->detection() )
    {
      detections.push_back( ts->detection() );
    }
  }

  return core::compute_average_classification(
    detections, weighted, scale_by_conf, ignore_class );
}


// ===============================================================================
write_object_track_set_viame_csv
::write_object_track_set_viame_csv()
  : d( new write_object_track_set_viame_csv::priv( this ) )
{
}


write_object_track_set_viame_csv
::~write_object_track_set_viame_csv()
{
}


void write_object_track_set_viame_csv
::close()
{
  if( d->m_active_writing )
  {
    // No flushing required
    write_object_track_set::close();
    return;
  }

  d->write_header_info( stream() );

  for( auto trk_pair : d->m_tracks )
  {
    auto trk_ptr = trk_pair.second;

    const kwiver::vital::detected_object_type_sptr trk_average_tot =
          ( d->m_tot_option == "detection" ? kwiver::vital::detected_object_type_sptr()
            : compute_average_tot( trk_ptr,
                d->m_tot_option.find( "weighted" ) != std::string::npos,
                d->m_tot_option.find( "scaled_by_conf" ) != std::string::npos,
                d->m_tot_ignore_class ) );

    for( auto ts_ptr : *trk_ptr )
    {
      kwiver::vital::object_track_state* ts =
        dynamic_cast< kwiver::vital::object_track_state* >( ts_ptr.get() );

      if( !ts )
      {
        LOG_ERROR( d->m_logger,
          "Invalid timestamp " << trk_ptr->id() << " " << trk_ptr->size() );
        continue;
      }

      kwiver::vital::detected_object_sptr det = ts->detection();
      const kwiver::vital::bounding_box_d empty_box =
        kwiver::vital::bounding_box_d( -1, -1, -1, -1 );
      kwiver::vital::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );
      auto confidence = ( det ? det->confidence() : 0 );
      kwiver::vital::frame_id_t frame_id = ts->frame() + d->m_frame_id_adjustment;

      stream() << trk_ptr->id() << d->m_delim            // 1: track id
               << d->format_image_id( ts ) << d->m_delim // 2: video or image id
               << frame_id << d->m_delim                 // 3: frame number
               << bbox.min_x() << d->m_delim             // 4: TL-x
               << bbox.min_y() << d->m_delim             // 5: TL-y
               << bbox.max_x() << d->m_delim             // 6: BR-x
               << bbox.max_y() << d->m_delim             // 7: BR-y
               << confidence << d->m_delim               // 8: confidence
               << "0";                                   // 9: length

      if( det )
      {
        const kwiver::vital::detected_object_type_sptr dot =
          ( d->m_tot_option == "detection" ? det->type() : trk_average_tot );

        if( dot )
        {
          for( auto name : dot->top_class_names( d->m_top_n_classes ) )
          {
            stream() << d->m_delim << name << d->m_delim << dot->score( name );
          }
        }

        d->write_detection_info( stream(), det );

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }

  write_object_track_set::close();
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_delim =
    config->get_value< std::string >( "delimiter", d->m_delim );
  d->m_stream_identifier =
    config->get_value< std::string >( "stream_identifier", d->m_stream_identifier );
  d->m_model_identifier =
    config->get_value< std::string >( "model_identifier", d->m_model_identifier );
  d->m_version_identifier =
    config->get_value< std::string >( "version_identifier", d->m_version_identifier );
  d->m_frame_rate =
    config->get_value< std::string >( "frame_rate", d->m_frame_rate );
  d->m_active_writing =
    config->get_value< bool >( "active_writing", d->m_active_writing );
  d->m_write_time_as_uid =
    config->get_value< bool >( "write_time_as_uid", d->m_write_time_as_uid );
  d->m_tot_option =
    config->get_value< std::string> ( "tot_option", d->m_tot_option );
  d->m_tot_ignore_class =
    config->get_value< std::string >( "tot_ignore_class", d->m_tot_ignore_class );
  d->m_frame_id_adjustment =
    config->get_value< int >( "frame_id_adjustment", d->m_frame_id_adjustment );
  d->m_top_n_classes =
    config->get_value< unsigned >( "top_n_classes", d->m_top_n_classes );
  d->m_mask_to_poly_tol =
      config->get_value< double >( "mask_to_poly_tol", d->m_mask_to_poly_tol );
  d->m_mask_to_poly_points =
      config->get_value< int >( "mask_to_poly_points", d->m_mask_to_poly_points );

  if( !d->m_active_writing )
  {
    time( &d->m_start_time );
  }
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


// -------------------------------------------------------------------------------
bool
write_object_track_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::write_set( const kwiver::vital::object_track_set_sptr& set,
             const kwiver::vital::timestamp& ts,
             const std::string& file_id )
{
  if( d->m_first )
  {
    if( d->m_active_writing )
    {
      d->write_header_info( stream() );
    }

    d->m_first = false;
  }

  if( !file_id.empty() && ts.has_valid_frame() )
  {
    d->m_frame_uids[ static_cast<unsigned>( ts.get_frame() ) ] = file_id;
  }

  if( !set )
  {
    return;
  }

  if( !d->m_active_writing )
  {
    for( auto trk : set->tracks() )
    {
      d->m_tracks[ static_cast<unsigned>( trk->id() ) ] = trk;
    }
  }
  else
  {
    for( auto trk_ptr : set->tracks() )
    {
      if( !trk_ptr || trk_ptr->empty() )
      {
        LOG_ERROR( d->m_logger, "Received invalid track" );
        continue;
      }

      kwiver::vital::object_track_state* state =
        dynamic_cast< kwiver::vital::object_track_state* >( trk_ptr->back().get() );

      if( !state )
      {
        LOG_ERROR( d->m_logger, "Invalid track state for track "
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

      kwiver::vital::detected_object_sptr det = state->detection();

      const kwiver::vital::bounding_box_d empty_box =
        kwiver::vital::bounding_box_d( -1, -1, -1, -1 );

      kwiver::vital::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );

      auto confidence = ( det ? det->confidence() : 0 );
      kwiver::vital::frame_id_t frame_id = state->frame() + d->m_frame_id_adjustment;

      stream() << trk_ptr->id() << d->m_delim               // 1: track id
               << d->format_image_id( state ) << d->m_delim // 2: video or image id
               << frame_id << d->m_delim                    // 3: frame number
               << bbox.min_x() << d->m_delim                // 4: TL-x
               << bbox.min_y() << d->m_delim                // 5: TL-y
               << bbox.max_x() << d->m_delim                // 6: BR-x
               << bbox.max_y() << d->m_delim                // 7: BR-y
               << confidence << d->m_delim                  // 8: confidence
               << "0";                                      // 9: length

      if( det )
      {
        const kwiver::vital::detected_object_type_sptr dot =
          ( d->m_tot_option == "detection" ? det->type() :
            compute_average_tot( trk_ptr,
              d->m_tot_option.find( "weighted" ) != std::string::npos,
              d->m_tot_option.find( "scaled_by_conf" ) != std::string::npos,
              d->m_tot_ignore_class ) );

        if( dot )
        {
          for( auto name : dot->top_class_names( d->m_top_n_classes ) )
          {
            stream() << d->m_delim << name << d->m_delim << dot->score( name );
          }
        }

        d->write_detection_info( stream(), det );

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }
}

} // end namespace viame
