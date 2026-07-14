/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header for train_detector_svm algorithm
 */

#ifndef VIAME_SVM_TRAIN_DETECTOR_SVM_H
#define VIAME_SVM_TRAIN_DETECTOR_SVM_H

#include "viame_svm_export.h"

#include <vital/algo/train_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <map>
#include <memory>
#include <string>

namespace viame {

namespace kv = kwiver::vital;

/// Train SVM models for object detection using descriptor-based classification
///
/// This algorithm trains binary SVM classifiers for each object category using
/// libsvm. It reads descriptor vectors from a CSV index file and trains models
/// based on positive/negative label files.
class VIAME_SVM_EXPORT train_detector_svm
  : public kv::algo::train_detector
{
public:
#define VIAME_SVM_TDS_PARAMS \
    PARAM_DEFAULT( \
      descriptor_index_file, std::string, \
      "Path to CSV file containing descriptor index (uid,v1,v2,...)", \
      "descriptors.csv" ), \
    PARAM_DEFAULT( \
      label_folder, std::string, \
      "Folder containing label files with descriptor UIDs per category", \
      "database" ), \
    PARAM_DEFAULT( \
      label_extension, std::string, \
      "File extension for label files", \
      "lbl" ), \
    PARAM_DEFAULT( \
      output_directory, std::string, \
      "Output directory for trained SVM model files", \
      "category_models" ), \
    PARAM_DEFAULT( \
      background_category, std::string, \
      "Category name to skip (background class)", \
      "background" ), \
    PARAM_DEFAULT( \
      maximum_positive_count, unsigned, \
      "Maximum number of positive samples per category", \
      75 ), \
    PARAM_DEFAULT( \
      maximum_negative_count, unsigned, \
      "Maximum number of negative samples per category", \
      750 ), \
    PARAM_DEFAULT( \
      minimum_positive_threshold, unsigned, \
      "Minimum number of positive samples required to train a category model", \
      1 ), \
    PARAM_DEFAULT( \
      minimum_negative_threshold, unsigned, \
      "Minimum number of negative samples required to train a category model", \
      1 ), \
    PARAM_DEFAULT( \
      train_on_neighbors_only, bool, \
      "If true, only use nearest-neighbor hard negatives. " \
      "If false, use 50/50 split of NN hard negatives and random negatives.", \
      false ), \
    PARAM_DEFAULT( \
      two_stage_training, bool, \
      "If true, train initial SVM to find hard negatives, then retrain. " \
      "If false, use single-stage training with NN-based hard negatives.", \
      true ), \
    PARAM_DEFAULT( \
      svm_rerank_negatives, bool, \
      "If true (default), train an intermediate SVM to re-rank and select hard negatives. " \
      "If false, use only centroid similarity for hard negative selection. " \
      "Only applies when two_stage_training is true.", \
      true ), \
    PARAM_DEFAULT( \
      auto_compute_neighbors, bool, \
      "If true, automatically compute pos_seed_neighbors as " \
      "maximum_negative_count / maximum_positive_count. " \
      "If false, use the manually specified pos_seed_neighbors value.", \
      true ), \
    PARAM_DEFAULT( \
      pos_seed_neighbors, unsigned, \
      "Number of nearest neighbors per positive sample for hard negative mining. " \
      "Only used if auto_compute_neighbors is false.", \
      10 ), \
    PARAM_DEFAULT( \
      min_pos_seed_neighbors, unsigned, \
      "Minimum number of neighbors per positive sample. " \
      "Enforced even when auto_compute_neighbors is true.", \
      2 ), \
    PARAM_DEFAULT( \
      feedback_sample_count, unsigned, \
      "Number of uncertain samples (near decision boundary) to include as additional " \
      "hard negatives. Set to 0 to disable.", \
      0 ), \
    PARAM_DEFAULT( \
      svm_kernel_type, std::string, \
      "SVM kernel type: linear, poly, rbf, sigmoid.", \
      "linear" ), \
    PARAM_DEFAULT( \
      svm_c, double, \
      "SVM regularization parameter C", \
      1.0 ), \
    PARAM_DEFAULT( \
      svm_gamma, double, \
      "SVM gamma parameter for rbf/poly/sigmoid kernels", \
      0.001 ), \
    PARAM_DEFAULT( \
      normalize_descriptors, bool, \
      "If true (default), L2-normalize descriptor vectors before training and scoring. " \
      "This is recommended for most descriptor types.", \
      true ), \
    PARAM_DEFAULT( \
      score_normalization, std::string, \
      "How to convert SVM output to [0,1] scores: " \
      "'sigmoid' (default) uses 1/(1+exp(-decision_value)), " \
      "'probability' uses libsvm's probability estimates (requires more training time).", \
      "sigmoid" ), \
    PARAM_DEFAULT( \
      use_class_weights, bool, \
      "If true, weight SVM classes inversely proportional to their frequency " \
      "to handle class imbalance", \
      false ), \
    PARAM_DEFAULT( \
      distance_metric, std::string, \
      "Distance metric for nearest neighbor search: euclidean or cosine", \
      "euclidean" ), \
    PARAM_DEFAULT( \
      random_seed, int, \
      "Random seed for reproducibility. Use -1 for random initialization.", \
      0 ), \
    PARAM_DEFAULT( \
      nn_index_type, std::string, \
      "Nearest neighbor index type: 'brute_force' (default) or 'lsh'. " \
      "LSH uses ITQ locality-sensitive hashing for faster approximate NN search.", \
      "brute_force" ), \
    PARAM_DEFAULT( \
      lsh_model_dir, std::string, \
      "Directory containing ITQ model files from generate_nn_index.py." \
      "Required when nn_index_type is 'lsh'.", \
      "" ), \
    PARAM_DEFAULT( \
      lsh_bit_length, unsigned, \
      "Number of bits in the ITQ hash code. Must match the trained model.", \
      256 ), \
    PARAM_DEFAULT( \
      lsh_itq_iterations, unsigned, \
      "Number of ITQ iterations used when training the model.", \
      100 ), \
    PARAM_DEFAULT( \
      lsh_random_seed, int, \
      "Random seed used when training the ITQ model.", \
      0 ), \
    PARAM_DEFAULT( \
      lsh_hash_ratio, double, \
      "Fraction of hash codes to search when using LSH. " \
      "Higher values give better recall but slower search. " \
      "Default is 0.2 (search top 20% of hash codes by Hamming distance).", \
      0.2 )

  PLUGGABLE_VARIABLES( VIAME_SVM_TDS_PARAMS )
  PLUGGABLE_CONSTRUCTOR( train_detector_svm, VIAME_SVM_TDS_PARAMS )

  static std::string plugin_name() { return "svm"; }
  static std::string plugin_description() { return "Train SVM models for object detection"; }

  PLUGGABLE_STATIC_FROM_CONFIG( train_detector_svm, VIAME_SVM_TDS_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_SVM_TDS_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( train_detector_svm, VIAME_SVM_TDS_PARAMS )

  /// Destructor
  virtual ~train_detector_svm();

  /// Check that the algorithm's configuration is valid
  virtual bool check_configuration( kv::config_block_sptr config ) const;

  /// Train a detection model given a list of images and detections
  virtual void
  add_data_from_disk(
    kv::category_hierarchy_sptr object_labels,
    std::vector< std::string > train_image_names,
    std::vector< kv::detected_object_set_sptr > train_groundtruth,
    std::vector< std::string > test_image_names = std::vector< std::string >(),
    std::vector< kv::detected_object_set_sptr > test_groundtruth
      = std::vector< kv::detected_object_set_sptr >() );

  /// Train all SVM models
  virtual std::map<std::string, std::string> update_model() override;

private:
  void initialize() override;
  void set_configuration_internal( kv::config_block_sptr config ) override;

  /// Private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace viame

#endif // VIAME_SVM_TRAIN_DETECTOR_SVM_H
