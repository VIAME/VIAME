/*ckwg +5
 * Copyright 2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#include <string>
#include <set>
#include <utility>

#include <track_oracle/track_scorable_mgrs/scorable_mgrs.h>

#include <track_oracle/core/track_oracle_instantiation.h>
#include <track_oracle/core/track_field_instantiation.h>
#include <track_oracle/core/track_oracle_row_view_instantiation.h>
#include <track_oracle/core/element_store_instantiation.h>
#include <track_oracle/core/kwiver_io_base_instantiation.h>

#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(T) \
  TRACK_ORACLE_INSTANCES(T) \
  TRACK_FIELD_INSTANCES_OLD_STYLE_DEFAULT_OUTPUT(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)

TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(kwiver::track_oracle::scorable_mgrs);
