// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <track_oracle/vibrant_descriptors/descriptor_cutic_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_metadata_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_motion_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_overlap_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_event_label_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_raw_1d_type.h>

#define TRACK_FIELD_EXPORT VIBRANT_DESCRIPTORS_EXPORT
#define KWIVER_IO_EXPORT VIBRANT_DESCRIPTORS_EXPORT
#define TRACK_ORACLE_CORE_EXPORT VIBRANT_DESCRIPTORS_EXPORT
#define ELEMENT_STORE_EXPORT VIBRANT_DESCRIPTORS_EXPORT
#define TRACK_ORACLE_ROW_VIEW_EXPORT VIBRANT_DESCRIPTORS_EXPORT

#include <track_oracle/core/track_oracle_instantiation.h>
#include <track_oracle/core/track_field_instantiation.h>
#include <track_oracle/core/track_oracle_row_view_instantiation.h>
#include <track_oracle/core/element_store_instantiation.h>
#include <track_oracle/core/kwiver_io_base_instantiation.h>

/// Shouldn't need to distinguish between these, but VS9 has a bug:
/// http://connect.microsoft.com/VisualStudio/feedback/details/753981

#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_ORACLE_INSTANCES(T)   \
  TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)

TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_cutic_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_metadata_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_motion_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_overlap_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_event_label_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_raw_1d_type);

#undef TRACK_ORACLE_ROW_VIEW_EXPORT
#undef ELEMENT_STORE_EXPORT
#undef TRACK_ORACLE_CORE_EXPORT
#undef KWIVER_IO_EXPORT
#undef TRACK_FIELD_EXPORT
