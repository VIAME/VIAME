/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_XGTF_H
#define INCL_TRACK_XGTF_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_xgtf/track_xgtf_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

#include <vgl/vgl_box_2d.h>

#include <string>
#include <utility>

namespace kwiver {
namespace track_oracle {

  /// This is the track_oracle schema for an XGTF track generated
  /// by the VIPER ground-truthing tool, used on e.g. VIRAT.
  /// The particular columns ("activity", "occlusion", etc.) are specific
  /// to MITRE's ground-truthing effort on VIRAT.

struct TRACK_XGTF_EXPORT track_xgtf_type: public track_base< track_xgtf_type >
{

  //track level data
  track_field< dt::tracking::external_id > external_id; //attribute id in phase one
  track_field< std::string >& type; // 'name' in phase one. could be pvo as well.
  track_field< std::pair< unsigned int, unsigned int > >& frame_span;

  //For activities, allowing string printed. Alternative is enum??
  track_field< int >& activity;
  track_field< double >& activity_probability;

  //frame level data
  track_field< dt::tracking::bounding_box > bounding_box;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< double >& occlusion;

  track_xgtf_type():

    /// track-level data
    type(Track.add_field< std::string >( "type") ),
    frame_span( Track.add_field< std::pair< unsigned int, unsigned int > >("frame_span") ),

    /// the index of the activity; in VIRAT, it corresponds to the indices
    /// found in aries_interface
    activity( Track.add_field< int >("activity") ),

    /// the probability of the activity; for ground-truth, it's fixed at 1.0
    activity_probability( Track.add_field< double >("activity_probability") ),

    /// frame-level data
    occlusion( Frame.add_field< double >("occlusion") )
  {
    Track.add_field( external_id );
    Frame.add_field( bounding_box );
    Frame.add_field( frame_number );
  }
};

} // ...track_oracle
} // ...kwiver

#endif
