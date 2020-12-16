// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_oracle_format_base_export.h>

#include <track_oracle/file_formats/file_format_manager.h>

#define TRACK_FIELD_EXPORT TRACK_ORACLE_FORMAT_BASE_EXPORT
#define KWIVER_IO_EXPORT TRACK_ORACLE_FORMAT_BASE_EXPORT
#define TRACK_ORACLE_CORE_EXPORT TRACK_ORACLE_FORMAT_BASE_EXPORT
#define ELEMENT_STORE_EXPORT TRACK_ORACLE_FORMAT_BASE_EXPORT
#define TRACK_ORACLE_ROW_VIEW_EXPORT TRACK_ORACLE_FORMAT_BASE_EXPORT

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

TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(kwiver::track_oracle::file_format_enum);
