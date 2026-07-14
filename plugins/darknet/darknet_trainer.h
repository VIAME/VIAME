/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_DARKNET_TRAINER_H
#define VIAME_DARKNET_TRAINER_H

#include "viame_darknet_export.h"

#include <vital/algo/train_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <map>
#include <string>

namespace viame {

class VIAME_DARKNET_EXPORT darknet_trainer :
  public kwiver::vital::algo::train_detector
{
public:
#define VIAME_DARKNET_DT_PARAMS \
    PARAM_DEFAULT( \
      net_config, std::string, \
      "Name of network config file.", \
      "" ), \
    PARAM_DEFAULT( \
      seed_weights, std::string, \
      "Optional input seed weights file.", \
      "" ), \
    PARAM_DEFAULT( \
      train_directory, std::string, \
      "Temp directory for all files used in training.", \
      "deep_training" ), \
    PARAM_DEFAULT( \
      output_directory, std::string, \
      "Final directory to output all models to.", \
      "category_models" ), \
    PARAM_DEFAULT( \
      output_model_name, std::string, \
      "Optional model name over-ride, if unspecified default used.", \
      "yolo" ), \
    PARAM_DEFAULT( \
      pipeline_template, std::string, \
      "Optional output kwiver pipeline for this detector", \
      "" ), \
    PARAM_DEFAULT( \
      model_type, std::string, \
      "Type of model (values understood are \"yolov2\" and \"yolov3\" [the " \
      "default]).", \
      "yolov3" ), \
    PARAM_DEFAULT( \
      skip_format, bool, \
      "Skip file formatting, assume that the train_directory is pre-populated " \
      "with all files required for model training.", \
      false ), \
    PARAM_DEFAULT( \
      gpu_index, int, \
      "GPU index. Only used when darknet is compiled with GPU support.", \
      0 ), \
    PARAM_DEFAULT( \
      resize_option, std::string, \
      "Pre-processing resize option, can be: disabled, maintain_ar, scale, " \
      "chip, or chip_and_original.", \
      "maintain_ar" ), \
    PARAM_DEFAULT( \
      scale, double, \
      "Image scaling factor used when resize_option is scale or chip.", \
      1.0 ), \
    PARAM_DEFAULT( \
      resize_width, int, \
      "Width resolution after resizing", \
      0 ), \
    PARAM_DEFAULT( \
      resize_height, int, \
      "Height resolution after resizing", \
      0 ), \
    PARAM_DEFAULT( \
      chip_step, int, \
      "When in chip mode, the chip step size between chips.", \
      100 ), \
    PARAM_DEFAULT( \
      overlap_required, double, \
      "Percentage of which a target must appear on a chip for it to be included " \
      "as a training sample for said chip.", \
      0.05 ), \
    PARAM_DEFAULT( \
      random_int_shift, double, \
      "Random intensity shift to add to each extracted chip [0.0,1.0].", \
      0.0 ), \
    PARAM_DEFAULT( \
      gs_to_rgb, bool, \
      "Convert input greyscale images to rgb before processing.", \
      true ), \
    PARAM_DEFAULT( \
      chips_w_gt_only, bool, \
      "Only chips with valid groundtruth objects on them will be included in " \
      "training.", \
      false ), \
    PARAM_DEFAULT( \
      max_neg_ratio, double, \
      "Do not use more than this many more frames without groundtruth in " \
      "training than there are frames with truth.", \
      0.0 ), \
    PARAM_DEFAULT( \
      ignore_category, std::string, \
      "Ignore this category in training, but still include chips around it.", \
      "false_alarm" ), \
    PARAM_DEFAULT( \
      min_train_box_length, int, \
      "If a box resizes to smaller than this during training, the input frame " \
      "will not be used in training.", \
      5 ), \
    PARAM_DEFAULT( \
      batch_size, int, \
      "Number of images per batch (and thus how many images constitute an iteration)", \
      64 ), \
    PARAM_DEFAULT( \
      batch_subdivisions, int, \
      "Number of subdivisions to split a batch into (thereby saving memory)", \
      16 )

  PLUGGABLE_VARIABLES( VIAME_DARKNET_DT_PARAMS )
  PLUGGABLE_CONSTRUCTOR( darknet_trainer, VIAME_DARKNET_DT_PARAMS )

  static std::string plugin_name() { return "darknet"; }
  static std::string plugin_description() { return "Training utility for darknet."; }

  PLUGGABLE_STATIC_FROM_CONFIG( darknet_trainer, VIAME_DARKNET_DT_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_DARKNET_DT_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( darknet_trainer, VIAME_DARKNET_DT_PARAMS )

  virtual ~darknet_trainer();

  virtual kwiver::vital::config_block_sptr get_configuration() const override;
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void add_data_from_disk(
    kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual void add_data_from_memory(
    kwiver::vital::category_hierarchy_sptr object_labels,
    std::vector< kwiver::vital::image_container_sptr > train_images,
    std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
    std::vector< kwiver::vital::image_container_sptr > test_images,
    std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth );

  virtual std::map<std::string, std::string> update_model() override;

private:
  void initialize() override;
  void set_configuration_internal( kwiver::vital::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif /* VIAME_DARKNET_TRAINER_H */
