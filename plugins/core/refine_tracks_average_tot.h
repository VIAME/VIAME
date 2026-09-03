/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_REFINE_TRACKS_AVERAGE_TOT_H
#define VIAME_CORE_REFINE_TRACKS_AVERAGE_TOT_H

#include "viame_core_export.h"

#include <vital/algo/refine_tracks.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <map>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class refine_tracks_average_tot
 *
 * \brief Replaces each track state's classification with a track-wide average
 *
 * Performs the same averaging the viame_csv track writer applies via its
 * tot_option parameter, but upstream of the writer so every output format
 * -- csv, coco, kw18, dive -- sees the same track object type.
 *
 * The average needs every state of a track, so it is computed in finalize()
 * over the accumulated stream and emitted as a single trailing batch. Writers
 * key tracks by id and keep the last version they are handed, so that batch
 * supersedes the per-frame sets passed through unmodified by refine().
 */
class VIAME_CORE_EXPORT refine_tracks_average_tot
  : public kwiver::vital::algo::refine_tracks
{
public:
  PLUGGABLE_IMPL_NAMED(
    refine_tracks_average_tot, "average_tot",
    "Replaces the classification on every state of a track with an average "
    "computed over the whole track.\n\n"
    "Mirrors the tot_option parameter of the viame_csv track writer, applied "
    "before the writer so that all output formats agree.",
    PARAM_DEFAULT(
      tot_option, std::string,
      "Track object type option: detection, average, weighted_average, "
      "weighted_average_scaled_by_conf. 'detection' disables averaging.",
      "weighted_average" ),
    PARAM_DEFAULT(
      tot_ignore_class, std::string,
      "Class name to ignore when computing the track object type average",
      "" )
  )

  virtual ~refine_tracks_average_tot() = default;

  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const;

  /// Record the frame's tracks and pass them through unmodified
  virtual kwiver::vital::object_track_set_sptr
  refine( kwiver::vital::timestamp ts,
          kwiver::vital::image_container_sptr image_data,
          kwiver::vital::object_track_set_sptr tracks ) const;

  /// Emit every accumulated track with its averaged classification applied
  virtual kwiver::vital::object_track_set_sptr
  finalize() const;

private:
  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  /// Latest version of each track seen, keyed by track id
  mutable std::map< kwiver::vital::track_id_t,
                    kwiver::vital::track_sptr > m_tracks;
};

} // end namespace

#endif // VIAME_CORE_REFINE_TRACKS_AVERAGE_TOT_H
