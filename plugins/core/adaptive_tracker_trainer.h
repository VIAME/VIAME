/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H
#define VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H

#include "viame_core_export.h"

#include <vital/algo/train_tracker.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Adaptive tracker trainer that analyzes data and runs appropriate training pipelines
 *
 * This algorithm analyzes tracking data characteristics (track counts, lengths,
 * motion patterns, fragmentation) and runs multiple configured tracker training
 * pipelines sequentially. Each trainer can have hard requirements (must be met
 * to run) and soft preferences (used for ranking when multiple trainers qualify).
 *
 * Statistics computed:
 * - Track counts and lengths (total, per-class, short/medium/long distribution)
 * - Track fragmentation (gaps, fragments per track)
 * - Motion patterns (velocity statistics, direction changes)
 * - Object density per frame (mean, max, crowded/sparse)
 * - Concurrent track counts (how many tracks active simultaneously)
 * - Occlusion analysis (track overlap/proximity)
 * - Object sizes (for Re-ID crop sizing decisions)
 * - Appearance consistency (bounding box size variance within tracks)
 *
 * Use cases:
 * - Running parameter-based trackers (ByteTrack) for simple motion scenarios
 * - Running Re-ID-based trackers (DeepSORT) when appearance features are needed
 * - Running advanced trackers (SRNN) for complex multi-object scenarios
 */
class VIAME_CORE_EXPORT adaptive_tracker_trainer
  : public kwiver::vital::algo::train_tracker
{
public:
#define VIAME_CORE_ATT_PARAMS \
    PARAM_DEFAULT( \
      max_trainers_to_run, size_t, \
      "Maximum number of trainers to run sequentially.", \
      3 ), \
    PARAM_DEFAULT( \
      short_track_threshold, size_t, \
      "Track length (frames) below which tracks are 'short'.", \
      10 ), \
    PARAM_DEFAULT( \
      long_track_threshold, size_t, \
      "Track length (frames) above which tracks are 'long'.", \
      100 ), \
    PARAM_DEFAULT( \
      stationary_velocity_threshold, double, \
      "Velocity (pixels/frame) below which objects are 'stationary'.", \
      2.0 ), \
    PARAM_DEFAULT( \
      fast_velocity_threshold, double, \
      "Velocity (pixels/frame) above which objects are 'fast'.", \
      50.0 ), \
    PARAM_DEFAULT( \
      sparse_frame_threshold, size_t, \
      "Max concurrent tracks for 'sparse' classification.", \
      3 ), \
    PARAM_DEFAULT( \
      crowded_frame_threshold, size_t, \
      "Min concurrent tracks for 'crowded' classification.", \
      15 ), \
    PARAM_DEFAULT( \
      small_object_threshold, double, \
      "Area threshold (pixels^2) below which objects are 'small'.", \
      1024.0 ), \
    PARAM_DEFAULT( \
      large_object_threshold, double, \
      "Area threshold (pixels^2) above which objects are 'large'.", \
      16384.0 ), \
    PARAM_DEFAULT( \
      close_distance_threshold, double, \
      "Distance (pixels) below which tracks are 'close'.", \
      50.0 ), \
    PARAM_DEFAULT( \
      high_variance_threshold, double, \
      "Size CV above which a track has 'high variance'.", \
      0.3 ), \
    PARAM_DEFAULT( \
      output_statistics_file, std::string, \
      "Optional file path for JSON statistics. Empty = disabled.", \
      "" ), \
    PARAM_DEFAULT( \
      verbose, bool, \
      "Enable verbose logging.", \
      true )

  PLUGGABLE_VARIABLES( VIAME_CORE_ATT_PARAMS )
  PLUGGABLE_CONSTRUCTOR( adaptive_tracker_trainer, VIAME_CORE_ATT_PARAMS )
  PLUGGABLE_IMPL_BASIC( adaptive_tracker_trainer, "Analyzes tracking data and runs appropriate training pipelines" )
  PLUGGABLE_STATIC_FROM_CONFIG( adaptive_tracker_trainer, VIAME_CORE_ATT_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_CORE_ATT_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( adaptive_tracker_trainer, VIAME_CORE_ATT_PARAMS )

  virtual ~adaptive_tracker_trainer() = default;

  virtual kwiver::vital::config_block_sptr get_configuration() const override;
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  virtual void
  add_data_from_disk( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::object_track_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::object_track_set_sptr > test_groundtruth ) override;

  virtual void
  add_data_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::object_track_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::object_track_set_sptr > test_groundtruth ) override;

  virtual void update_model() override;

private:
  void initialize() override;
  void set_configuration_internal( kwiver::vital::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif /* VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H */
