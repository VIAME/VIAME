// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief The track_oracle base class for KPF geometry files.
 *
 * We differentiate between geometry and activity files because the
 * alternative, i.e. treating a KPF like a CSV or kwiver-xml file,
 * would require infrastructure to resolve linkages between activities
 * (which refer to, but do not contain, geometry) and the reference geometry.
 *
 * (We've never tried using CSV or kwiver-xml for activities; it would
 * probably not work as-is.)
 *
 */

#ifndef KWIVER_TRACK_ORACLE_TRACK_KPF_GEOM_H_
#define KWIVER_TRACK_ORACLE_TRACK_KPF_GEOM_H_

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kpf_geom/track_kpf_geom_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KPF_GEOM_EXPORT track_kpf_geom_type: public track_base< track_kpf_geom_type >
{
  track_field< dt::detection::detection_id > det_id;
  track_field< dt::tracking::external_id > track_id;
  track_field< dt::tracking::timestamp_usecs > timestamp_usecs;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< dt::tracking::bounding_box > bounding_box;

  track_kpf_geom_type()
  {
    Track.add_field( track_id );
    Frame.add_field( det_id );
    Frame.add_field( timestamp_usecs );
    Frame.add_field( frame_number );
    Frame.add_field( bounding_box );
  }
};

} // ...track_oracle
} // ...kwiver

#endif
