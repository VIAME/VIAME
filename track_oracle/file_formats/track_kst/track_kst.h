// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_KST_H
#define INCL_TRACK_KST_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kst/track_kst_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>
#include <string>
#include <utility>
#include <vgl/vgl_box_2d.h>

namespace kwiver {
namespace track_oracle {

  /// This is the track_oracle schema for a KST track generated
  /// by viqui.  At the moment, this only reads in columns pertinent
  /// to scoring.

struct TRACK_KST_EXPORT track_kst_type: public track_base< track_kst_type >
{
  track_field< dt::virat::descriptor_classifier > descriptor_classifier;
  track_field< unsigned >& instance_id;
  track_field< double >& relevancy;
  track_field< unsigned >& rank;

  //Frame specific
  track_field< vgl_box_2d< double > >& bounding_box;
  track_field< unsigned >& frame_number;
  track_field< unsigned long long>& timestamp_usecs;

  track_kst_type():
    instance_id( Track.add_field< unsigned >( "instance_id" )),
    relevancy( Track.add_field< double >( "relevancy" )),
    rank( Track.add_field< unsigned >( "rank" )),
    bounding_box(Frame.add_field< vgl_box_2d< double > >("bounding_box")),
    frame_number(Frame.add_field< unsigned >("frame_number")),
    timestamp_usecs(Frame.add_field< unsigned long long >("timestamp_usecs"))
  {
    Track.add_field( descriptor_classifier );
  }

};

} // ...track_oracle
} // ...kwiver

#endif
