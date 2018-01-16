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
 * @brief The track_oracle filter for KPF activities.
 *
 * Like KWE, a KPF activity file:
 *
 * - does not contain any geometry; instead it acts as a "track view" into
 *   the actor tracks. Since we don't yet support track views, we clone
 *   the source tracks.
 *
 * - Does not support the generic read() interface, instead requiring the
 *   source tracks to have been read already
 *
 *
 */

#ifndef INCL_TRACK_FILTER_KPF_ACTIVITY_H
#define INCL_TRACK_FILTER_KPF_ACTIVITY_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_filter_kpf_activity/track_filter_kpf_activity_export.h>

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_FILTER_KPF_ACTIVITY_EXPORT track_filter_kpf_activity:
    public track_base< track_filter_kpf_activity >
{

  track_field< dt::events::event_id > activity_id;
  track_field< kpf_cset_type >& activity_labels;
  track_field< dt::events::kpf_activity_domain > activity_domain;
  track_field< dt::events::actor_track_rows > actors;
  track_field< dt::events::kpf_activity_start > activity_start;
  track_field< dt::events::kpf_activity_stop > activity_stop;

  track_filter_kpf_activity():
    activity_labels( Track.add_field< kpf_cset_type >( "kpf_activity_labels" ))
  {
    Track.add_field( activity_id );
    Track.add_field( activity_domain );
    Track.add_field( actors );
    Track.add_field( activity_start );
    Track.add_field( activity_stop );
  };

  static bool read( const std::string& fn,
                    const track_handle_list_type& ref_tracks,
                    int kpf_activity_domain,
                    track_handle_list_type& new_tracks );

  static bool write( const std::string& fn,
                     const track_handle_list_type& tracks );
};

} // ...track_oracle
} // ...kwiver


#endif
