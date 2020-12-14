// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_ORACLE_ROW_VIEW_INSTANCES_H
#define INCL_TRACK_ORACLE_ROW_VIEW_INSTANCES_H

#include <track_oracle/core/track_oracle_row_view.txx>

#define TRACK_ORACLE_ROW_VIEW_INSTANCES(T) \
  template TRACK_ORACLE_ROW_VIEW_EXPORT kwiver::track_oracle::track_field< T >& kwiver::track_oracle::track_oracle_row_view::add_field< T >( const std::string& );

#endif
