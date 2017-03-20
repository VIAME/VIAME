/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FILTER_KWE_H
#define INCL_TRACK_FILTER_KWE_H

#include <vital/vital_config.h>
#include <track_oracle/track_filter_kwe/track_filter_kwe_export.h>

#include <track_oracle/event_adapter.h>

/// This schema actually is the event_adapter.h::event_data_schema
/// with a read() method added on to it.
///
/// This is the first example of a "track filter", which is
/// information contained in a file that refers to tracks, adds
/// information to tracks, but does not itself contain track geometry.
/// This gets handled specially because loading this file by itself
/// will NOT result in tracks.  In particular, you need to have
/// already loaded the tracks that the filter refers to.
///
/// This particular filter is the legacy event format for Kitware
/// event detectors.  It is an exceptionally brittle format, mostly
/// because the event type is encoded as an arbitrary enum over in
/// event_types.h .
///
/// Other peculiarities of KWE files and scoring:
///
/// -- KWE files can record vidtk events, but we (at the moment) we
/// only score VIRAT events.  The conversion is handled in
/// aries_interface.
///
/// -- KWEs do not contain geometry but instead references to subsets
/// of other tracks, i.e. they are track views.  Since track_oracle
/// doesn't yet support track views, we clone the subsets of the
/// source tracks into new tracks, based on the track IDs.  When we clone,
/// we clone all the non-system fields.
///
/// -- Since this is a filter, and not a format, it's not handled by
/// the file format manager's generic read() interface.  Instead,
/// supporting the clone operation described above requires passing
/// in the set of tracks which the KWE refers to.  (This could also
/// be handled by domains.)
///

namespace kwiver {
namespace track_oracle {

struct TRACK_FILTER_KWE_EXPORT track_filter_kwe_type: public event_data_schema
{
  static bool read( const std::string& fn,
                    const track_handle_list_type& ref_tracks,
                    const std::string& track_style,
                    track_handle_list_type& new_tracks );
};

} // ...track_oracle
} // ...kwiver


#endif
