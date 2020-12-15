/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_KWXML_H
#define INCL_TRACK_KWXML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kwxml/track_kwxml_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>
#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <track_oracle/vibrant_descriptors/descriptor_cutic_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_metadata_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_motion_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_overlap_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_event_label_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_raw_1d_type.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KWXML_EXPORT track_kwxml_type: public track_base< track_kwxml_type >
{

  //Track specific
  track_field< dt::tracking::external_id > external_id;
  track_field< unsigned >& video_id;
  track_field< std::string >& time_stamp;
  track_field< dt::events::source_track_ids> source_track_ids;
  track_field< std::string >& track_style;

  // part of the great Track Unification Effort
  // ...xgtf
  track_field< int >& activity;
  track_field< double >& activity_probability;
  // ...kst
  track_field< double >& relevancy;

  //Frame specific
  track_field< std::string >& type;
  track_field< dt::tracking::bounding_box > bounding_box;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< dt::tracking::timestamp_usecs > timestamp_usecs;

  //Descriptors

  track_field< dt::virat::descriptor_classifier > descriptor_classifier;
  track_field< std::vector< std::vector< double > > >& descriptor_uthof;
  track_field< std::vector< double > >& descriptor_rpidbn1;
  track_field< std::vector< double > >& descriptor_rpidbn2;
  track_field< std::vector< double > >& descriptor_kwsoti3;
  track_field< std::vector< double > >& descriptor_icsihog;
  track_field< std::vector< double > >& descriptor_texashog;
  track_field< std::vector< double > >& descriptor_cornell;
  track_field< std::vector< double > >& descriptor_pvo_raw_scores;
  track_field< std::vector< double > >& descriptor_umdlds;
  track_field< std::vector< double > >& descriptor_umd_periodicity;
  track_field< std::vector< double > >& descriptor_umdssm;
  //Skipping CUTIC...
  track_field< descriptor_cutic_type >& descriptor_cutic;
  track_field< std::vector< double > >& descriptor_cubof;
  track_field< std::vector< double > >& descriptor_cu_texture;
  track_field< std::vector< double > >& descriptor_cu_col_moment;
  track_field< descriptor_overlap_type >& descriptor_overlap;
  track_field< descriptor_event_label_type>& descriptor_event_label;
  track_field< descriptor_raw_1d_type>& descriptor_raw_1d;
  track_field< double >& descriptor_query_result_score;
  //Skipping metadataDescriptor
  track_field< descriptor_metadata_type >& descriptor_metadata;
  //Skipping motionDescriptor
  track_field< descriptor_motion_type >& descriptor_motion;

  // e2at
  track_field< std::string >& clip_filename;
  track_field< double >& start_time_secs;
  track_field< double >& end_time_secs;
  track_field< std::string >& basic_annotation;
  track_field< std::string >& augmented_annotation;
  track_field< double >& latitude;
  track_field< double >& longitude;


  track_kwxml_type():
    video_id( Track.add_field< unsigned >("video_id")),
    time_stamp( Track.add_field< std::string >("time_stamp_str")),
    track_style(Track.add_field< std::string >("track_style")),

    activity(Track.add_field< int >("activity")),
    activity_probability(Track.add_field< double >("activity_probability")),
    relevancy( Track.add_field< double >( "relevancy" )),

    type(Frame.add_field< std::string >("type")),  //should be bool raw?

    descriptor_uthof(Track.add_field< std::vector< std::vector< double > > >("descriptor_uthof")),
    descriptor_rpidbn1(Track.add_field< std::vector< double > >("descriptor_rpidbn1")),
    descriptor_rpidbn2(Track.add_field< std::vector< double > >("descriptor_rpidbn2")),
    descriptor_kwsoti3(Track.add_field< std::vector< double > >("descriptor_kwsoti3")),
    descriptor_icsihog(Track.add_field< std::vector< double > >("descriptor_icsihog")),
    descriptor_texashog(Track.add_field< std::vector< double > >("descriptor_texashog")),
    descriptor_cornell(Track.add_field< std::vector< double > >("descriptor_cornell")),
    descriptor_pvo_raw_scores(Track.add_field< std::vector< double > >("descriptor_pvo_raw_scores")),
    descriptor_umdlds(Track.add_field< std::vector< double > >("descriptor_umdlds")),
    descriptor_umd_periodicity(Track.add_field< std::vector< double > >("descriptor_umd_periodicity")),
    descriptor_umdssm(Track.add_field< std::vector< double > >("descriptor_umdssm")),
    descriptor_cutic(Track.add_field< descriptor_cutic_type >("descriptor_cutic")),
    descriptor_cubof(Track.add_field< std::vector< double > >("descriptor_cubof")),
    descriptor_cu_texture(Track.add_field< std::vector< double > >("descriptor_cu_texture")),
    descriptor_cu_col_moment(Track.add_field< std::vector< double > >("descriptor_cu_col_moment")),
    descriptor_overlap( Track.add_field< descriptor_overlap_type >( "descriptor_overlap" )),
    descriptor_event_label( Track.add_field< descriptor_event_label_type >( "descriptor_event_label" )),
    descriptor_raw_1d( Track.add_field< descriptor_raw_1d_type >( "descriptor_raw_1d" )),
    descriptor_query_result_score(Track.add_field< double >("descriptor_query_result_score")),
    descriptor_metadata(Track.add_field< descriptor_metadata_type >("descriptor_metadata")),
    descriptor_motion(Track.add_field< descriptor_motion_type >("descriptor_motion")),

    clip_filename( Track.add_field< std::string >( "clip_filename" )),
    start_time_secs( Track.add_field< double >( "start_time_secs" )),
    end_time_secs( Track.add_field< double >( "end_time_secs" )),
    basic_annotation( Track.add_field< std::string >( "basic_annotation" )),
    augmented_annotation( Track.add_field< std::string >( "augmented_annotation" )),
    latitude( Track.add_field< double >( "latitude" )),
    longitude( Track.add_field< double >( "longitude" ))

  {
    Track.add_field( external_id );
    Track.add_field( source_track_ids );
    Track.add_field( descriptor_classifier );

    Frame.add_field( bounding_box );
    Frame.add_field( frame_number );
    Frame.add_field( timestamp_usecs );
  }
};

} // ...track_oracle
} // ...kwiver

#endif
