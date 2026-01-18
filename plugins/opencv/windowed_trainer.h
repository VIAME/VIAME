/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef VIAME_OPENCV_WINDOWED_TRAINER_H
#define VIAME_OPENCV_WINDOWED_TRAINER_H

#include "viame_opencv_export.h"

#include <vital/algo/train_detector.h>
#include <vital/algo/image_io.h>
#include <vital/algo/algorithm.txx>
#include <vital/plugin_management/pluggable_macro_magic.h>
#include <vital/types/bounding_box.h>

#include "windowed_utils.h"

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Run training on arbitrary other trainers in a windowed fashion
 */
class VIAME_OPENCV_EXPORT windowed_trainer
  : public kwiver::vital::algo::train_detector
{
public:
  PLUGGABLE_IMPL(
    windowed_trainer,
    "Window some other arbitrary detector trainer across the image",
    PARAM_DEFAULT(
      mode, std::string,
      "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
      "chip, chip_and_original, original_and_resized, or adaptive.",
      "disabled" ),
    PARAM_DEFAULT(
      scale, double,
      "Image scaling factor used when mode is scale or chip.",
      1.0 ),
    PARAM_DEFAULT(
      chip_width, int,
      "When in chip mode, the chip width.",
      1000 ),
    PARAM_DEFAULT(
      chip_height, int,
      "When in chip mode, the chip height.",
      1000 ),
    PARAM_DEFAULT(
      chip_step_width, int,
      "When in chip mode, the chip step size between chips.",
      500 ),
    PARAM_DEFAULT(
      chip_step_height, int,
      "When in chip mode, the chip step size between chips.",
      500 ),
    PARAM_DEFAULT(
      chip_edge_filter, int,
      "If using chipping, filter out detections this pixel count near borders.",
      -1 ),
    PARAM_DEFAULT(
      chip_edge_max_prob, double,
      "If using chipping, maximum type probability for edge detections",
      -1.0 ),
    PARAM_DEFAULT(
      chip_adaptive_thresh, int,
      "If using adaptive selection, total pixel count at which we start to chip.",
      2000000 ),
    PARAM_DEFAULT(
      batch_size, int,
      "Optional processing batch size to send to the detector.",
      1 ),
    PARAM_DEFAULT(
      min_detection_dim, int,
      "Minimum detection dimension in original image space.",
      1 ),
    PARAM_DEFAULT(
      original_to_chip_size, bool,
      "Optionally enforce the input image is the specified chip size",
      false ),
    PARAM_DEFAULT(
      black_pad, bool,
      "Black pad the edges of resized chips to ensure consistent dimensions",
      false ),
    PARAM_DEFAULT(
      train_directory, std::string,
      "Directory for all files used in training.",
      "deep_training" ),
    PARAM_DEFAULT(
      chip_format, std::string,
      "Image format for output chips.",
      "png" ),
    PARAM_DEFAULT(
      skip_format, bool,
      "Skip file formatting, assume that the train_directory is pre-populated "
      "with all files required for model training.",
      false ),
    PARAM_DEFAULT(
      chip_random_factor, double,
      "A percentage [0.0, 1.0] of chips to randomly use in training",
      -1.0 ),
    PARAM_DEFAULT(
      always_write_image, bool,
      "Always re-write images to training directory even if they already exist "
      "elsewhere on disk.",
      false ),
    PARAM_DEFAULT(
      ensure_standard, bool,
      "If images are not one of 3 common formats (jpg, jpeg, png) or 3 channel "
      "write them to the training directory even if they are elsewhere already",
      false ),
    PARAM_DEFAULT(
      overlap_required, double,
      "Percentage of which a target must appear on a chip for it to be included "
      "as a training sample for said chip.",
      0.05 ),
    PARAM_DEFAULT(
      chips_w_gt_only, bool,
      "Only chips with valid groundtruth objects on them will be included in "
      "training.",
      false ),
    PARAM_DEFAULT(
      max_neg_ratio, double,
      "Do not use more than this many more frames without groundtruth in "
      "training than there are frames with truth.",
      0.0 ),
    PARAM_DEFAULT(
      random_validation, double,
      "Randomly add this percentage of training frames to validation.",
      0.0 ),
    PARAM_DEFAULT(
      ignore_category, std::string,
      "Ignore this category in training, but still include chips around it.",
      "false_alarm" ),
    PARAM_DEFAULT(
      min_train_box_length, int,
      "If a box resizes to smaller than this during training, the input frame "
      "will not be used in training.",
      0 ),
    PARAM_DEFAULT(
      min_train_box_edge_dist, double,
      "If non-zero and a box is within a chip boundary adjusted by this many "
      "pixels, do not train on the chip.",
      0.0 ),
    PARAM_DEFAULT(
      small_box_area, int,
      "If a box resizes to smaller than this during training, consider it a small "
      "detection which might lead to several modifications to it.",
      0 ),
    PARAM_DEFAULT(
      small_action, std::string,
      "Action to take in the event that a detection is considered small. Can "
      "either be none, remove, or any other string which will over-ride the "
      "detection type to be that string.",
      "" ),
    PARAM(
      image_reader, kwiver::vital::algo::image_io_sptr,
      "Algorithm pointer to nested image reader" ),
    PARAM(
      trainer, kwiver::vital::algo::train_detector_sptr,
      "Algorithm pointer to nested trainer" )
  )

  virtual ~windowed_trainer() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;



  virtual void
  add_data_from_disk( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void
  add_data_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void update_model();

private:
  // Runtime state
  kwiver::vital::category_hierarchy_sptr m_labels;
  std::map< std::string, int > m_category_map;
  bool m_synthetic_labels = true;

  // Helper functions
  void format_images_from_disk(
    const window_settings& settings,
    std::vector< std::string > image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > groundtruth,
    std::vector< std::string >& formatted_names,
    std::vector< kwiver::vital::detected_object_set_sptr >& formatted_truth );

  void format_image_from_memory(
    const window_settings& settings,
    const cv::Mat& image,
    kwiver::vital::detected_object_set_sptr groundtruth,
    const rescale_option format_method,
    std::vector< std::string >& formatted_names,
    std::vector< kwiver::vital::detected_object_set_sptr >& formatted_truth );

  bool filter_detections_in_roi(
    kwiver::vital::detected_object_set_sptr all_detections,
    kv::bounding_box_d region,
    kv::detected_object_set_sptr& filt_detections );

  std::string generate_filename( const int len = 10 );

  void write_chip_to_disk( const std::string& filename, const cv::Mat& image );

  const std::string m_chip_subdirectory = "cached_chips";
};


} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_TRAINER_H */
