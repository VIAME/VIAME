/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shared stereo track pairing utilities
 *
 * Provides the stereo_track_pairer class and helpers used by both
 * pair_stereo_detections_process and measure_objects_process.
 */

#ifndef VIAME_CORE_PAIR_STEREO_TRACKS_H
#define VIAME_CORE_PAIR_STEREO_TRACKS_H

#include "viame_core_export.h"

#include <vital/vital_types.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/detected_object.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>
#include <vital/config/config_block.h>

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <utility>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

// =============================================================================
// Helper structs for track accumulation
// =============================================================================

struct id_pair
{
  kv::track_id_t left_id;
  kv::track_id_t right_id;
};

struct pairing
{
  std::set< kv::frame_id_t > frame_set;
  id_pair left_right_id_pair;
};

struct split_range
{
  kv::track_id_t left_id, right_id, new_track_id;
  kv::frame_id_t frame_id_first, frame_id_last;
  int detection_count;
};

// =============================================================================
// Union-Find for transitive stereo track association.
// Left IDs are stored as-is (non-negative).
// Right IDs are encoded as -(right_id + 1) to avoid collision with left ID 0.
// =============================================================================

class VIAME_CORE_EXPORT track_union_find
{
public:
  kv::track_id_t find( kv::track_id_t x );
  void unite( kv::track_id_t a, kv::track_id_t b );
  std::map< kv::track_id_t, std::set< kv::track_id_t > > groups();

private:
  std::map< kv::track_id_t, kv::track_id_t > parent;
  std::map< kv::track_id_t, int > rnk;
};

// =============================================================================
// Free helper functions for stereo class averaging
// =============================================================================

VIAME_CORE_EXPORT kv::detected_object_type_sptr
compute_stereo_average_classification(
  const std::vector< kv::detected_object_sptr >& dets_left,
  const std::vector< kv::detected_object_sptr >& dets_right,
  bool weighted,
  bool scale_by_conf = false,
  const std::string& ignore_class = "" );

VIAME_CORE_EXPORT void
apply_classification_to_track(
  const kv::track_sptr& trk,
  const kv::detected_object_type_sptr& dot );

VIAME_CORE_EXPORT void
average_track_lengths(
  const kv::track_sptr& trk1,
  const kv::track_sptr& trk2,
  double iqr_factor );

// =============================================================================
// Main class: stereo_track_pairer
// =============================================================================

class VIAME_CORE_EXPORT stereo_track_pairer
{
public:
  stereo_track_pairer();
  ~stereo_track_pairer();

  // Config (same pattern as map_keypoints_to_camera_settings)
  kv::config_block_sptr get_configuration() const;
  void set_configuration( kv::config_block_sptr config );

  // Accessors
  bool accumulation_enabled() const;
  bool output_unmatched() const;
  bool average_stereo_classes() const;
  bool use_weighted_averaging() const;
  bool use_scaled_by_conf() const;
  std::string class_averaging_ignore_class() const;
  kv::track_id_t allocate_track_id();

  // Per-frame track remapping with union-find + class averaging
  void remap_tracks_per_frame(
    const kv::object_track_set_sptr& tracks1,
    const kv::object_track_set_sptr& tracks2,
    const std::vector< std::pair< int, int > >& matches,
    const std::vector< kv::track_id_t >& track_ids1,
    const std::vector< kv::track_id_t >& track_ids2,
    std::vector< kv::track_sptr >& output1,
    std::vector< kv::track_sptr >& output2 );

  // Accumulation mode
  void accumulate_frame_pairings(
    const std::vector< std::pair< int, int > >& matches,
    const std::vector< kv::detected_object_sptr >& detections1,
    const std::vector< kv::detected_object_sptr >& detections2,
    const std::vector< kv::track_id_t >& track_ids1,
    const std::vector< kv::track_id_t >& track_ids2,
    const kv::timestamp& timestamp );

  void resolve_accumulated_pairings(
    std::vector< kv::track_sptr >& output1,
    std::vector< kv::track_sptr >& output2 );

private:
  // Config
  bool m_accumulate_track_pairings = false;
  std::string m_pairing_resolution_method = "most_likely";
  int m_detection_split_threshold = 3;
  int m_min_track_length = 0;
  int m_max_track_length = 0;
  double m_min_avg_surface_area = 0.0;
  double m_max_avg_surface_area = 0.0;
  bool m_average_stereo_classes = false;
  bool m_average_stereo_lengths = false;
  double m_length_outlier_iqr_factor = 1.5;
  std::string m_class_averaging_method = "weighted_average";
  std::string m_class_averaging_ignore_class;
  bool m_output_unmatched = true;

  // State
  track_union_find m_track_union_find;
  std::map< kv::track_id_t, kv::track_id_t > m_left_to_output_id;
  std::map< kv::track_id_t, kv::track_id_t > m_right_to_output_id;
  kv::track_id_t m_next_track_id = 0;

  // Accumulation state
  std::map< kv::track_id_t, kv::track_sptr > m_accumulated_tracks1;
  std::map< kv::track_id_t, kv::track_sptr > m_accumulated_tracks2;
  std::map< size_t, pairing > m_left_to_right_pairing;

  // Private helpers
  static size_t cantor_pairing_fn( size_t i, size_t j );

  void select_most_likely_pairing(
    std::vector< kv::track_sptr >& left_tracks,
    std::vector< kv::track_sptr >& right_tracks,
    std::set< kv::track_id_t >& proc_left,
    std::set< kv::track_id_t >& proc_right );

  void split_paired_tracks(
    std::vector< kv::track_sptr >& left_tracks,
    std::vector< kv::track_sptr >& right_tracks,
    std::set< kv::track_id_t >& proc_left,
    std::set< kv::track_id_t >& proc_right );

  std::vector< split_range > create_split_ranges_from_track_pairs() const;

  std::vector< kv::track_sptr > filter_tracks(
    std::vector< kv::track_sptr > tracks ) const;

  kv::track_id_t last_accumulated_track_id() const;
};

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_PAIR_STEREO_TRACKS_H
