/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_ORACLE_FRAME_VIEW_H
#define INCL_TRACK_ORACLE_FRAME_VIEW_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_oracle_row_view.h>

namespace kwiver {
namespace track_oracle {

class TRACK_ORACLE_EXPORT track_oracle_frame_view: public track_oracle_row_view
{
  friend class track_base_impl;

private:
  track_oracle_row_view& parent_track_view;

  bool unlink( const oracle_entry_handle_type& row );

  frame_handle_type create();

public:
  explicit track_oracle_frame_view( track_oracle_row_view& parent );

  const track_oracle_frame_view& operator[]( frame_handle_type row ) const;

};

} // ...track_oracle
} // ...kwiver

#endif
