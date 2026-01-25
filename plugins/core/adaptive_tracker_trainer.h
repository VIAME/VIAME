/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H
#define VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H

#include "viame_core_export.h"

#include <vital/algo/train_tracker.h>

#include <map>
#include <string>

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
  : public kwiver::vital::algorithm_impl< adaptive_tracker_trainer,
      kwiver::vital::algo::train_tracker >
{
public:

  PLUGIN_INFO( "adaptive",
               "Analyzes tracking data and runs appropriate training pipelines" )

  adaptive_tracker_trainer();
  virtual ~adaptive_tracker_trainer();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void
  add_data_from_disk( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::object_track_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::object_track_set_sptr > test_groundtruth );

  virtual void
  add_data_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::object_track_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::object_track_set_sptr > test_groundtruth );

  virtual std::map<std::string, std::string> update_model() override;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_CORE_ADAPTIVE_TRACKER_TRAINER_H */
