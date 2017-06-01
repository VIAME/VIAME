/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_BASE_H
#define INCL_TRACK_BASE_H

#include <track_oracle/core/track_base_impl.h>

namespace kwiver {
namespace track_oracle {

template <typename client_derived_from_track_base_type,
          typename track_base_type = track_base_impl >
class track_base: public track_base_type
{
public:

  // Syntactic sugar for frame access - thanks Amitha!
  // ...not frame access any more; calling t(xx) is implicitly the track
  client_derived_from_track_base_type& operator()( const track_handle_type& t )
  {
    this->Track.set_cursor( t.row );
    return static_cast<client_derived_from_track_base_type&>(*this);
  }
  client_derived_from_track_base_type& operator[]( const frame_handle_type& f )
  {
    this->Frame.set_cursor( f.row );
    return static_cast<client_derived_from_track_base_type&>(*this);
  }

};

} // ...kwiver
} // ...track_oracle

#endif
