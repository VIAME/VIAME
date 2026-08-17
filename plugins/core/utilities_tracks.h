/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_TRACKS_H
#define VIAME_CORE_UTILITIES_TRACKS_H

#include "viame_core_export.h"

#include <vital/types/detected_object.h>
#include <vital/types/object_track_set.h>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Track resampling utilities
// =============================================================================

/// Linearly interpolate a detection between two states
///
/// Bounding box and confidence are interpolated; the classification and any
/// auxiliary data (polygons, masks, notes) are taken from the nearer input.
///
/// \param d1 Detection at the earlier state
/// \param d2 Detection at the later state
/// \param alpha Interpolation weight in [0, 1] (0 = d1, 1 = d2)
/// \returns New interpolated detection
VIAME_CORE_EXPORT
kv::detected_object_sptr
interpolate_detection( const kv::detected_object_sptr& d1,
                       const kv::detected_object_sptr& d2,
                       double alpha );

/// Resample object tracks from one frame rate to another
///
/// Rescales every track state's frame number from the input rate to the
/// output rate (frame f becomes f * output_rate / input_rate) and creates
/// states at all output frames within each track's temporal extent. Output
/// frames that align with an annotated input state reuse that detection
/// unmodified; frames in between are filled by linear interpolation between
/// the surrounding annotated states (or by copying the nearest state when
/// \p interpolate_states is false). Track extents are never extrapolated.
/// The output rate may be higher or lower than the input rate.
///
/// State times are set to frame / output_rate. Tracks whose whole extent
/// falls between two output frames snap to the nearest output frame so that
/// single-state tracks survive downsampling.
///
/// \param tracks Input track set with frame numbers at \p input_rate
/// \param input_rate Frame rate (Hz) the input annotations were made at
/// \param output_rate Desired output frame rate (Hz)
/// \param interpolate_states Interpolate between annotated states instead of
///                           copying the nearest one
/// \param max_interp_gap Maximum gap between consecutive annotated states,
///                       in input frames, across which new states are
///                       created (0 = unlimited)
/// \returns New track set with frame numbers at \p output_rate
VIAME_CORE_EXPORT
kv::object_track_set_sptr
resample_object_tracks( const kv::object_track_set_sptr& tracks,
                        double input_rate,
                        double output_rate,
                        bool interpolate_states = true,
                        unsigned max_interp_gap = 0 );

} // end namespace viame

#endif // VIAME_CORE_UTILITIES_TRACKS_H
