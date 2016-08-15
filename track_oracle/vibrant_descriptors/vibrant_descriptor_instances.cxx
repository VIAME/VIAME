/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#include <track_oracle/vibrant_descriptors/descriptor_cutic_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_metadata_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_motion_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_overlap_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_event_label_type.h>
#include <track_oracle/vibrant_descriptors/descriptor_raw_1d_type.h>

#include <track_oracle/track_oracle_instantiation.h>
#include <track_oracle/track_field_instantiation.h>
#include <track_oracle/track_field_functor_instantiation.h>
#include <track_oracle/track_oracle_row_view_instantiation.h>
#include <track_oracle/element_store_instantiation.h>
#include <track_oracle/kwiver_io_base_instantiation.h>


/// Shouldn't need to distinguish between these, but VS9 has a bug:
/// http://connect.microsoft.com/VisualStudio/feedback/details/753981

#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_ORACLE_INSTANCES(T)   \
  TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_FIELD_FUNCTOR_INSTANCES(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)

TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_cutic_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_metadata_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_motion_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_overlap_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_event_label_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::descriptor_raw_1d_type);

