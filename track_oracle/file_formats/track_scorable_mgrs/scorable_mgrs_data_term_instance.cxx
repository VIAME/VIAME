// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <track_oracle/track_scorable_mgrs/scorable_mgrs_data_term.h>

#include <track_oracle/track_oracle_instantiation.h>
#include <track_oracle/track_field_instantiation.h>
#include <track_oracle/track_field_functor_instantiation.h>
#include <track_oracle/track_oracle_row_view_instantiation.h>
#include <track_oracle/element_store_instantiation.h>
#include <track_oracle/kwiver_io_base_instantiation.h>

#define TRACK_ORACLE_INSTANTIATE_DATA_TERM(T) \
  TRACK_FIELD_INSTANCES_DATA_TERM(T) \
  KWIVER_IO_BASE_INSTANCES(T)

TRACK_ORACLE_INSTANTIATE_DATA_TERM( ::kwiver::track_oracle::dt::tracking::mgrs_pos );
