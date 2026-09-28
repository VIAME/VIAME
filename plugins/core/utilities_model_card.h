/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_MODEL_CARD_H
#define VIAME_CORE_UTILITIES_MODEL_CARD_H

#include "viame_core_export.h"

#include <vital/types/category_hierarchy.h>
#include <vital/types/detected_object_set.h>

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace viame {

/// Training metadata and final data splits used to describe a trained model.
struct model_card_inputs
{
  std::string output_directory;
  std::string detector_pipeline;         // Empty when no detector was trained
  std::string tracker_pipeline;          // Empty when no tracker was trained
  std::vector< std::string > detector_types;
  std::vector< std::string > tracker_types;
  std::string config_file;
  std::string init_weights;
  bool gt_frames_only = false;
  kwiver::vital::category_hierarchy_sptr labels;

  // Sequence-level split; every entry of `items` is train unless listed below.
  std::vector< std::string > items;
  std::vector< std::size_t > item_frame_counts;
  std::set< std::size_t > validation_items;   // Indices into items
  std::vector< std::string > test_items;

  // Frame-level split as actually handed to the trainer.
  std::vector< std::string > train_frames;
  std::vector< kwiver::vital::detected_object_set_sptr > train_truth;
  std::vector< std::string > validation_frames;
  std::vector< kwiver::vital::detected_object_set_sptr > validation_truth;
  bool validation_auto_selected = false;
  double validation_percent = 0.0;

  std::string test_results_dir;          // Empty when the test set was not scored
};

/// Write MODEL_CARD.md and the exact training/validation frame lists in splits/.
///
/// The output directory must already exist. If test_results_dir contains
/// summary.txt, its evaluation summary is included in the card.
/// Reports the output path or an inability to open the card on standard output.
VIAME_CORE_EXPORT
void write_model_card( const model_card_inputs& inputs );

} // namespace viame

#endif // VIAME_CORE_UTILITIES_MODEL_CARD_H
