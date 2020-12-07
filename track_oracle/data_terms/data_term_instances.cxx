// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <track_oracle/data_terms/data_terms.h>
#include <track_oracle/core/state_flags.h>

#include <vital/vital_config.h>
#include <track_oracle/data_terms/data_terms_export.h>
#define TRACK_FIELD_EXPORT DATA_TERMS_EXPORT
#define KWIVER_IO_EXPORT DATA_TERMS_EXPORT

#include <track_oracle/core/track_field_instantiation.h>
#include <track_oracle/core/kwiver_io_base_instantiation.h>

#define TRACK_ORACLE_INSTANTIATE_DATA_TERM(T) \
  TRACK_FIELD_INSTANCES_DATA_TERM(T) \
  KWIVER_IO_BASE_INSTANCES(T)

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::utility::state_flags);

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::detection::detection_id );

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::external_id );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::timestamp_usecs );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::frame_number );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::fg_mask_area );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::track_location );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::obj_x );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::obj_y );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::obj_location );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::velocity_x );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::velocity_y );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::bounding_box );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::world_x );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::world_y );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::world_z );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::world_location );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::latitude );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::longitude );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::time_stamp );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::world_gcs );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::track_uuid );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::track_style );

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::event_id );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::event_type );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::event_probability );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::source_track_ids );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::actor_track_rows );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::kpf_activity_domain );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::kpf_activity_start );
TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::events::kpf_activity_stop );

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::virat::descriptor_classifier );

template std::ostream& ::kwiver::track_oracle::operator<<(std::ostream& os, const kwiver::track_oracle::track_field_io_proxy< vgl_box_2d<double> > & );

#undef KWIVER_IO_EXPORT
#undef TRACK_FIELD_EXPORT
