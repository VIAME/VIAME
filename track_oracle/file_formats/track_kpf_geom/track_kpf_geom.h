/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
