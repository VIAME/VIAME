/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TRAINING_ADAPTIVE_DETECTOR_TRAINER_H
#define VIAME_TRAINING_ADAPTIVE_DETECTOR_TRAINER_H

#include "viame_training_export.h"

#include <viame/algorithm_framework/algo/train_detector.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

#include <map>
#include <string>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Adaptive detector trainer that analyzes data and runs appropriate training pipelines
 *
 * This algorithm analyzes training data characteristics (annotation counts,
 * object sizes) and runs multiple configured training pipelines sequentially.
 * Each trainer can have hard requirements (must be met to run) and soft
 * preferences (used for ranking when multiple trainers qualify).
 *
 * Use cases:
 * - Automatically running SVM for small datasets, deep learning for large
 * - Running tiled/windowed training for small objects
 * - Training multiple models with different characteristics
 */
class VIAME_TRAINING_EXPORT adaptive_detector_trainer
  : public viame::algo::train_detector
{
public:
#define VIAME_CORE_ADT_PARAMS \
    PARAM_DEFAULT( \
      max_trainers_to_run, size_t, \
      "Maximum number of trainers to run sequentially.", \
      3 ), \
    PARAM_DEFAULT( \
      small_object_threshold, double, \
      "Area threshold (pixels^2) below which objects are 'small'.", \
      1024.0 ), \
    PARAM_DEFAULT( \
      large_object_threshold, double, \
      "Area threshold (pixels^2) above which objects are 'large'.", \
      16384.0 ), \
    PARAM_DEFAULT( \
      tall_aspect_threshold, double, \
      "Aspect ratio (w/h) below which objects are 'tall'.", \
      0.5 ), \
    PARAM_DEFAULT( \
      wide_aspect_threshold, double, \
      "Aspect ratio (w/h) above which objects are 'wide'.", \
      2.0 ), \
    PARAM_DEFAULT( \
      low_annotation_threshold, size_t, \
      "Annotation count below which datasets are 'low' data.", \
      500 ), \
    PARAM_DEFAULT( \
      high_annotation_threshold, size_t, \
      "Annotation count above which datasets are 'high' data.", \
      2000 ), \
    PARAM_DEFAULT( \
      sparse_frame_threshold, size_t, \
      "Max objects/frame for 'sparse' classification.", \
      5 ), \
    PARAM_DEFAULT( \
      crowded_frame_threshold, size_t, \
      "Min objects/frame for 'crowded' classification.", \
      20 ), \
    PARAM_DEFAULT( \
      rare_class_threshold, size_t, \
      "Count below which a class is 'rare'.", \
      50 ), \
    PARAM_DEFAULT( \
      dominant_class_threshold, size_t, \
      "Count above which a class is 'dominant'.", \
      500 ), \
    PARAM_DEFAULT( \
      edge_margin_fraction, double, \
      "Fraction of image dimension for edge margin.", \
      0.05 ), \
    PARAM_DEFAULT( \
      overlap_iou_threshold, double, \
      "IoU threshold for 'high overlap' classification.", \
      0.3 ), \
    PARAM_DEFAULT( \
      output_statistics_file, std::string, \
      "Optional file path for JSON statistics. Empty = disabled.", \
      "" ), \
    PARAM_DEFAULT( \
      verbose, bool, \
      "Enable verbose logging.", \
      true )

  PLUGGABLE_VARIABLES( VIAME_CORE_ADT_PARAMS )
  PLUGGABLE_CONSTRUCTOR( adaptive_detector_trainer, VIAME_CORE_ADT_PARAMS )
  PLUGGABLE_IMPL_BASIC_NAMED( adaptive_detector_trainer, "adaptive", "Analyzes training data and runs appropriate training pipelines" )
  PLUGGABLE_STATIC_FROM_CONFIG( adaptive_detector_trainer, VIAME_CORE_ADT_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_CORE_ADT_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( adaptive_detector_trainer, VIAME_CORE_ADT_PARAMS )

  virtual ~adaptive_detector_trainer() = default;

  virtual viame::config_block_sptr get_configuration() const override;
  virtual bool check_configuration( viame::config_block_sptr config ) const override;

  virtual void
  add_data_from_disk( viame::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< viame::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< viame::detected_object_set_sptr > test_groundtruth );

  virtual void
  add_data_from_memory( viame::category_hierarchy_sptr object_labels,
    std::vector< viame::image_container_sptr > train_images,
    std::vector< viame::detected_object_set_sptr > train_groundtruth,
    std::vector< viame::image_container_sptr > test_images,
    std::vector< viame::detected_object_set_sptr > test_groundtruth );

  virtual std::map<std::string, std::string> update_model() override;

private:
  void initialize() override;
  void set_configuration_internal( viame::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif /* VIAME_TRAINING_ADAPTIVE_DETECTOR_TRAINER_H */
