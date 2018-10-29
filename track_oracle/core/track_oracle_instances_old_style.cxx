/*ckwg +5
 * Copyright 2014-2018 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#include <string>
#include <set>
#include <utility>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>

#include <vital/types/timestamp.h>
#include <vital/types/uid.h>

#include <track_oracle/core/state_flags.h>

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
#include <track_oracle/core/kwiver_io_base_instantiation.h>


#define TRACK_ORACLE_INSTANTIATE_DEBUG(T) \
  TRACK_ORACLE_INSTANCES(T) \
  TRACK_FIELD_INSTANCES_OLD_STYLE_DEFAULT_OUTPUT_DEBUG(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)

#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(T) \
  TRACK_ORACLE_INSTANCES(T) \
  TRACK_FIELD_INSTANCES_OLD_STYLE_DEFAULT_OUTPUT(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)

// passing types with a comma is tricky

#define MACRO_COMMA ,

TRACK_ORACLE_INSTANCES(kwiver::track_oracle::state_flag_type);
ELEMENT_STORE_INSTANCES(kwiver::track_oracle::state_flag_type);

TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(bool);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(int);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(double);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(unsigned int);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(unsigned long);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(unsigned long long);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(std::string);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(vgl_box_2d<unsigned>);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(vgl_box_2d<double>);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(vgl_point_2d<double>);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(vgl_point_3d<double>);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(kwiver::vital::timestamp);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(std::vector< kwiver::vital::timestamp >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_DEFAULT_OUTPUT(kwiver::vital::uid);

/// Shouldn't need to distinguish between these, but VS9 has a bug:
/// http://connect.microsoft.com/VisualStudio/feedback/details/753981

#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_ORACLE_INSTANCES(T)   \
  TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  ELEMENT_STORE_INSTANCES(T) \
  KWIVER_IO_BASE_INSTANCES(T)


#define TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT_COMMA(T, T2)     \
  TRACK_ORACLE_INSTANCES(T MACRO_COMMA T2) \
  TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT_COMMA(T, T2)           \
  TRACK_FIELD_INSTANCES_GENERAL(T MACRO_COMMA T2) \
  TRACK_ORACLE_ROW_VIEW_INSTANCES(T MACRO_COMMA T2) \
  ELEMENT_STORE_INSTANCES(T MACRO_COMMA T2) \
  KWIVER_IO_BASE_INSTANCES(T MACRO_COMMA T2)


TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::frame_handle_list_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::track_handle_list_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(std::vector< unsigned >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(std::vector< double >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(std::vector< std::vector< double > >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(std::vector< std::string >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(std::set< std::string >);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::kpf_cset_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::kpf_cset_sys_type);
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT(kwiver::track_oracle::kpf_cset_s2i_type);


#ifdef WIN32
TRACK_ORACLE_INSTANCES(std::pair<unsigned MACRO_COMMA unsigned >);
TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT(std::pair<unsigned MACRO_COMMA unsigned>);
TRACK_FIELD_INSTANCES_GENERAL(std::pair<unsigned MACRO_COMMA unsigned>);
TRACK_ORACLE_ROW_VIEW_INSTANCES(std::pair<unsigned MACRO_COMMA unsigned>);
ELEMENT_STORE_INSTANCES(std::pair<unsigned MACRO_COMMA unsigned>);
KWIVER_IO_BASE_INSTANCES(std::pair<unsigned MACRO_COMMA unsigned>);
#else
TRACK_ORACLE_INSTANTIATE_OLD_STYLE_SPECIAL_OUTPUT_COMMA(std::pair<unsigned, unsigned>);
#endif

#undef TRACK_ORACLE_ROW_VIEW_EXPORT
#undef ELEMENT_STORE_EXPORT
#undef TRACK_ORACLE_CORE_EXPORT
#undef KWIVER_IO_EXPORT
#undef TRACK_FIELD_EXPORT
