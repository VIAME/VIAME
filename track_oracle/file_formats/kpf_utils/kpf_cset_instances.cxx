// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Template instances of kwiver I/O for the CSET support types
 */

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>
#define TRACK_FIELD_EXPORT TRACK_ORACLE_EXPORT
#define KWIVER_IO_EXPORT TRACK_ORACLE_EXPORT
#define TRACK_ORACLE_CORE_EXPORT TRACK_ORACLE_EXPORT
#define ELEMENT_STORE_EXPORT TRACK_ORACLE_EXPORT
#define TRACK_ORACLE_ROW_VIEW_EXPORT TRACK_ORACLE_EXPORT

#include <track_oracle/core/track_oracle_instantiation.h>
#include <track_oracle/core/track_field_instantiation.h>
#include <track_oracle/core/track_oracle_row_view_instantiation.h>
#include <track_oracle/core/element_store_instantiation.h>

#define MACRO_COMMA ,

TRACK_ORACLE_INSTANCES(std::map<std::string MACRO_COMMA size_t>);
ELEMENT_STORE_INSTANCES(std::map<std::string MACRO_COMMA size_t>);
TRACK_ORACLE_INSTANCES(std::map<size_t MACRO_COMMA double>);
ELEMENT_STORE_INSTANCES(std::map<size_t MACRO_COMMA double>);

