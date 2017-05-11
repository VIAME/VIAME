/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_CALLOUT_H
#define INCL_TRACK_CALLOUT_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_e2at_callout/track_e2at_callout_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>

/*
This is a schema for the E2AT callout CSVs.
*/

namespace kwiver {
namespace track_oracle {

struct TRACK_E2AT_CALLOUT_EXPORT track_e2at_callout_type: public track_base< track_e2at_callout_type >
{
  // all track-level data
  track_field< std::string >& clip_filename;
  track_field< double >& start_time_secs;
  track_field< double >& end_time_secs;
  track_field< std::string >& basic_annotation;
  track_field< std::string >& augmented_annotation;
  track_field< double >& latitude;
  track_field< double >& longitude;

  track_e2at_callout_type():
    clip_filename( Track.add_field< std::string >( "clip_filename" )),
    start_time_secs( Track.add_field< double >( "start_time_secs" )),
    end_time_secs( Track.add_field< double >( "end_time_secs" )),
    basic_annotation( Track.add_field< std::string >( "basic_annotation" )),
    augmented_annotation( Track.add_field< std::string >( "augmented_annotation" )),
    latitude( Track.add_field< double >( "latitude" )),
    longitude( Track.add_field< double >( "longitude" ))
  {
  }
};

} // ...track_oracle
} // ...kwiver

#endif
