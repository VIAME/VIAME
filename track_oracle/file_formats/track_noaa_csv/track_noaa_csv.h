/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * @brief The track_oracle class for VIAME NOAA CSV files.
 *
 * Header consists of:
 * 1) detection / track ID
 * 2) video / image ID
 * 3) frame ID
 * 4) bbox top left x
 * 5) bbox top left y
 * 6) bbox bottom right x
 * 7) bbox bottom right y
 * 8) detection confidence
 * 9) fish length
 * 10) species #1 string
 * 11) species #1 confidence
 * ...10/11 repeat as necessary
 *
 * The (10/11) pairs are treated as a KPF cset10 packet.
 * Treat (8) as traditional relevance.
 */

#ifndef KWIVER_TRACK_ORACLE_TRACK_NOAA_CSV_H_
#define KWIVER_TRACK_ORACLE_TRACK_NOAA_CSV_H_

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_noaa_csv/track_noaa_csv_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_NOAA_CSV_EXPORT track_noaa_csv_type: public track_base< track_noaa_csv_type >
{
  track_field< dt::detection::detection_id > det_id;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< dt::tracking::bounding_box > bounding_box;
  track_field< kpf_cset_type >& species_cset;
  track_field< double >& relevancy;

  //
  // relevancy is nominally on the track
  //

  track_noaa_csv_type() :
    species_cset( Frame.add_field< kpf_cset_type >( "species_cset" )),
    relevancy( Track.add_field< double >( "relevancy" ))
  {
    Track.add_field( det_id );
    Frame.add_field( frame_number );
    Frame.add_field( bounding_box );
  }
};

} // ...track_oracle
} // ...kwiver

#endif
