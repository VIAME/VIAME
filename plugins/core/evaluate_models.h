// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/// \file
/// \brief Model evaluation utilities for computing detection and tracking metrics

#ifndef VIAME_CORE_EVALUATE_MODELS_H
#define VIAME_CORE_EVALUATE_MODELS_H

#include "viame_core_export.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace viame
{

// ----------------------------------------------------------------------------
/// \brief Configuration options for model evaluation
struct VIAME_CORE_EXPORT evaluation_config
{
  /// IoU threshold for matching detections to ground truth (default: 0.5)
  double iou_threshold = 0.5;

  /// Minimum confidence threshold for detections (default: 0.0, use all)
  double confidence_threshold = 0.0;

  /// Whether to compute tracking metrics (requires track IDs)
  bool compute_tracking_metrics = true;

  /// Whether to compute per-class metrics in addition to overall metrics
  bool compute_per_class_metrics = false;

  /// Frame tolerance for temporal matching (in frames, 0 = exact match)
  unsigned frame_tolerance = 0;
};

// ----------------------------------------------------------------------------
/// \brief Result structure containing all computed metrics
///
/// This structure organizes metrics into logical categories for easier access.
/// All metrics are also available in the combined `all_metrics` map.
struct VIAME_CORE_EXPORT evaluation_results
{
  /// Combined map of all metric names to values
  std::map< std::string, double > all_metrics;

  // -- Detection metrics --

  /// True positives count
  double true_positives = 0.0;
  /// False positives count
  double false_positives = 0.0;
  /// False negatives count (missed detections)
  double false_negatives = 0.0;
  /// Precision = TP / (TP + FP)
  double precision = 0.0;
  /// Recall = TP / (TP + FN)
  double recall = 0.0;
  /// F1 score = 2 * precision * recall / (precision + recall)
  double f1_score = 0.0;
  /// Matthews Correlation Coefficient
  double mcc = 0.0;
  /// Average Precision (area under PR curve)
  double average_precision = 0.0;
  /// AP at IoU threshold 0.5
  double ap50 = 0.0;
  /// AP at IoU threshold 0.75
  double ap75 = 0.0;
  /// AP averaged over IoU 0.5:0.95 (COCO-style mAP)
  double ap50_95 = 0.0;
  /// False Discovery Rate = FP / (FP + TP)
  double false_discovery_rate = 0.0;
  /// Miss Rate = FN / (FN + TP), complementary to recall
  double miss_rate = 0.0;

  // -- Localization quality metrics --

  /// Mean IoU of all true positive matches
  double mean_iou = 0.0;
  /// Median IoU of true positive matches
  double median_iou = 0.0;
  /// Mean center distance between matched boxes (pixels)
  double mean_center_distance = 0.0;
  /// Mean relative size error between matched boxes
  double mean_size_error = 0.0;

  // -- MOT tracking metrics --

  /// Multiple Object Tracking Accuracy
  double mota = 0.0;
  /// Multiple Object Tracking Precision (avg IoU on matches)
  double motp = 0.0;
  /// ID F1 score (global ID association quality)
  double idf1 = 0.0;
  /// ID Precision
  double idp = 0.0;
  /// ID Recall
  double idr = 0.0;
  /// Number of ID switches
  double id_switches = 0.0;
  /// Number of track fragmentations
  double fragmentations = 0.0;
  /// Mostly tracked objects (tracked >= 80% of lifespan)
  double mostly_tracked = 0.0;
  /// Partially tracked objects (tracked 20-80% of lifespan)
  double partially_tracked = 0.0;
  /// Mostly lost objects (tracked < 20% of lifespan)
  double mostly_lost = 0.0;

  // -- HOTA metrics (Higher Order Tracking Accuracy) --

  /// HOTA score (geometric mean of DetA and AssA, averaged over thresholds)
  double hota = 0.0;
  /// Detection Accuracy (balanced detection metric)
  double deta = 0.0;
  /// Association Accuracy (measures association quality)
  double assa = 0.0;
  /// Localization Accuracy (average IoU of true positives)
  double loca = 0.0;

  // -- KWANT-style tracking metrics --

  /// Average track continuity (1 = no breaks)
  double avg_track_continuity = 0.0;
  /// Average track purity (fraction dominated by single GT)
  double avg_track_purity = 0.0;
  /// Average target continuity
  double avg_target_continuity = 0.0;
  /// Average target purity
  double avg_target_purity = 0.0;
  /// Track probability of detection
  double track_pd = 0.0;
  /// Track false alarm rate
  double track_fa = 0.0;

  // -- Track quality metrics --

  /// Average length of computed tracks (in frames)
  double avg_track_length = 0.0;
  /// Average length of ground truth tracks (in frames)
  double avg_gt_track_length = 0.0;
  /// Average fraction of GT track covered by best matching computed track
  double track_completeness = 0.0;
  /// Average length of gaps within fragmented tracks
  double avg_gap_length = 0.0;

  // -- Normalized rates --

  /// Mostly tracked ratio (fraction of GT tracks)
  double mt_ratio = 0.0;
  /// Partially tracked ratio (fraction of GT tracks)
  double pt_ratio = 0.0;
  /// Mostly lost ratio (fraction of GT tracks)
  double ml_ratio = 0.0;
  /// False Alarms per Frame
  double faf = 0.0;

  // -- Classification metrics --

  /// Classification accuracy among true positives
  double classification_accuracy = 0.0;
  /// Mean AP across all classes
  double mean_ap = 0.0;

  // -- Additional statistics --

  /// Total number of ground truth objects
  double total_gt_objects = 0.0;
  /// Total number of computed detections
  double total_computed = 0.0;
  /// Total number of frames evaluated
  double total_frames = 0.0;
  /// Total unique ground truth track IDs
  double total_gt_tracks = 0.0;
  /// Total unique computed track IDs
  double total_computed_tracks = 0.0;

  // -- Per-class metrics (if enabled) --

  /// Per-class metric maps: class_name -> metric_name -> value
  std::map< std::string, std::map< std::string, double > > per_class_metrics;

  /// Populate the all_metrics map from individual fields
  void populate_all_metrics();
};

// ----------------------------------------------------------------------------
/// \brief A single point on a precision-recall curve
struct VIAME_CORE_EXPORT pr_curve_point
{
  double recall = 0.0;
  double precision = 0.0;
  double confidence = 0.0;  ///< Confidence threshold at this point
  double f1 = 0.0;          ///< F1 score at this point
  int tp = 0;               ///< True positives at this threshold
  int fp = 0;               ///< False positives at this threshold
  int fn = 0;               ///< False negatives at this threshold
};

// ----------------------------------------------------------------------------
/// \brief Precision-recall curve data for a single class or overall
struct VIAME_CORE_EXPORT pr_curve_data
{
  std::string class_name;   ///< Class name (empty for overall curve)
  std::vector< pr_curve_point > points;
  double average_precision = 0.0;  ///< Area under the PR curve
  double max_f1 = 0.0;             ///< Maximum F1 score on the curve
  double best_threshold = 0.0;     ///< Confidence threshold at max F1
};

// ----------------------------------------------------------------------------
/// \brief Confusion matrix data
struct VIAME_CORE_EXPORT confusion_matrix_data
{
  /// Ordered list of class names (row/column labels)
  std::vector< std::string > class_names;

  /// Matrix[i][j] = count of GT class i predicted as class j
  /// Last row/column is "background" (false negatives / false positives)
  std::vector< std::vector< int > > matrix;

  /// Normalized matrix (each row sums to 1.0)
  std::vector< std::vector< double > > normalized_matrix;

  /// Per-class accuracy (diagonal / row sum)
  std::map< std::string, double > per_class_accuracy;

  /// Overall accuracy (sum of diagonal / total)
  double overall_accuracy = 0.0;
};

// ----------------------------------------------------------------------------
/// \brief ROC curve point (for detection operating characteristic)
struct VIAME_CORE_EXPORT roc_curve_point
{
  double false_positive_rate = 0.0;  ///< FP / (FP + TN) or FP per image
  double true_positive_rate = 0.0;   ///< TP / (TP + FN) = recall
  double confidence = 0.0;
};

// ----------------------------------------------------------------------------
/// \brief ROC curve data
struct VIAME_CORE_EXPORT roc_curve_data
{
  std::string class_name;
  std::vector< roc_curve_point > points;
  double auc = 0.0;  ///< Area under the ROC curve
};

// ----------------------------------------------------------------------------
/// \brief Container for all plot data from an evaluation
struct VIAME_CORE_EXPORT evaluation_plot_data
{
  /// Overall precision-recall curve
  pr_curve_data overall_pr_curve;

  /// Per-class precision-recall curves
  std::map< std::string, pr_curve_data > per_class_pr_curves;

  /// Confusion matrix
  confusion_matrix_data confusion_matrix;

  /// Overall ROC curve (detection operating characteristic)
  roc_curve_data overall_roc_curve;

  /// Per-class ROC curves
  std::map< std::string, roc_curve_data > per_class_roc_curves;

  /// KWANT-style track purity histogram data
  /// Bins: [0-10%, 10-20%, ..., 90-100%] -> count of tracks
  std::vector< int > track_purity_histogram;

  /// KWANT-style track continuity histogram
  std::vector< int > track_continuity_histogram;

  /// Track length distribution histogram
  std::map< int, int > track_length_histogram;  ///< length -> count

  /// IoU distribution histogram for true positives
  std::vector< int > iou_histogram;  ///< Bins of 0.05 from 0 to 1
};

// ----------------------------------------------------------------------------
/// \brief Main evaluation class for computing detection and tracking metrics
///
/// This class evaluates computed detections/tracks against ground truth files
/// and produces comprehensive metrics including:
/// - Standard detection metrics (precision, recall, F1, MCC, AP)
/// - MOT metrics (MOTA, MOTP, IDF1, ID switches, fragmentations)
/// - HOTA metrics (HOTA, DetA, AssA, LocA)
/// - KWANT-style metrics (continuity, purity, Pd, FA)
///
/// Supported input formats:
/// - VIAME CSV format
///
/// Example usage:
/// \code
/// viame::model_evaluator evaluator;
/// evaluator.set_config(config);
///
/// std::vector<std::string> computed_files = {"detections.csv"};
/// std::vector<std::string> groundtruth_files = {"groundtruth.csv"};
///
/// auto results = evaluator.evaluate(computed_files, groundtruth_files);
/// for (const auto& metric : results.all_metrics)
/// {
///   std::cout << metric.first << ": " << metric.second << std::endl;
/// }
/// \endcode
class VIAME_CORE_EXPORT model_evaluator
{
public:
  /// Constructor
  model_evaluator();

  /// Destructor
  ~model_evaluator();

  /// Copy constructor (deleted - use move semantics)
  model_evaluator( const model_evaluator& ) = delete;

  /// Move constructor
  model_evaluator( model_evaluator&& ) noexcept;

  /// Copy assignment (deleted - use move semantics)
  model_evaluator& operator=( const model_evaluator& ) = delete;

  /// Move assignment
  model_evaluator& operator=( model_evaluator&& ) noexcept;

  /// \brief Set the evaluation configuration
  /// \param config Configuration options for evaluation
  void set_config( const evaluation_config& config );

  /// \brief Get the current evaluation configuration
  /// \returns Current configuration
  evaluation_config get_config() const;

  /// \brief Evaluate computed detections/tracks against ground truth
  ///
  /// This is the main evaluation function. It reads the input files,
  /// matches computed detections to ground truth, and computes all
  /// configured metrics.
  ///
  /// \param computed_files Vector of paths to computed detection/track files
  /// \param groundtruth_files Vector of paths to ground truth files
  /// \returns evaluation_results containing all computed metrics
  ///
  /// \note Files are paired by index - computed_files[i] is evaluated against
  ///       groundtruth_files[i]. The vectors must have the same size.
  evaluation_results evaluate(
    const std::vector< std::string >& computed_files,
    const std::vector< std::string >& groundtruth_files );

  /// \brief Get the results as a simple string-to-double map
  ///
  /// Convenience function that returns just the all_metrics map from
  /// the evaluation results.
  ///
  /// \param computed_files Vector of paths to computed detection/track files
  /// \param groundtruth_files Vector of paths to ground truth files
  /// \returns Map of metric names to values
  std::map< std::string, double > evaluate_to_map(
    const std::vector< std::string >& computed_files,
    const std::vector< std::string >& groundtruth_files );

  // -------------------------------------------------------------------------
  // Plotting data generation
  // -------------------------------------------------------------------------

  /// \brief Generate all plot data from the current evaluation
  ///
  /// Must be called after evaluate(). Generates PR curves, confusion matrix,
  /// ROC curves, and histogram data.
  ///
  /// \returns Container with all plot data
  evaluation_plot_data generate_plot_data();

  /// \brief Generate precision-recall curve for overall detections
  /// \param num_points Number of points on the curve (default 101 for 0-100%)
  /// \returns PR curve data with points sorted by recall
  pr_curve_data generate_pr_curve( int num_points = 101 );

  /// \brief Generate per-class precision-recall curves
  /// \param num_points Number of points per curve
  /// \returns Map of class name to PR curve data
  std::map< std::string, pr_curve_data > generate_per_class_pr_curves(
    int num_points = 101 );

  /// \brief Generate confusion matrix from matched detections
  /// \returns Confusion matrix with class labels and counts
  confusion_matrix_data generate_confusion_matrix();

  /// \brief Generate ROC curve (detection operating characteristic)
  /// \param num_points Number of points on the curve
  /// \returns ROC curve data
  roc_curve_data generate_roc_curve( int num_points = 101 );

  // -------------------------------------------------------------------------
  // Plot data export functions
  // -------------------------------------------------------------------------

  /// \brief Export PR curve data to CSV file
  /// \param curve The PR curve data to export
  /// \param filepath Output file path
  /// \returns true on success
  static bool export_pr_curve_csv( const pr_curve_data& curve,
                                   const std::string& filepath );

  /// \brief Export confusion matrix to CSV file
  /// \param matrix The confusion matrix to export
  /// \param filepath Output file path
  /// \returns true on success
  static bool export_confusion_matrix_csv( const confusion_matrix_data& matrix,
                                           const std::string& filepath );

  /// \brief Export all plot data to a directory
  ///
  /// Creates multiple files:
  /// - pr_curve_overall.csv
  /// - pr_curve_<class>.csv for each class
  /// - confusion_matrix.csv
  /// - roc_curve_overall.csv
  /// - histograms.csv
  ///
  /// \param plot_data The plot data to export
  /// \param output_dir Directory to write files to
  /// \returns true on success
  static bool export_plot_data( const evaluation_plot_data& plot_data,
                                const std::string& output_dir );

  /// \brief Export plot data to JSON format
  /// \param plot_data The plot data to export
  /// \param filepath Output JSON file path
  /// \returns true on success
  static bool export_plot_data_json( const evaluation_plot_data& plot_data,
                                     const std::string& filepath );

private:
  class priv;
  std::unique_ptr< priv > d;
};

// ----------------------------------------------------------------------------
/// \brief Convenience function to evaluate models without creating an evaluator
///
/// This function creates a temporary model_evaluator with the given config
/// and returns the results as a map.
///
/// \param computed_files Vector of paths to computed detection/track files
/// \param groundtruth_files Vector of paths to ground truth files
/// \param config Evaluation configuration (optional, uses defaults if not provided)
/// \returns Map of metric names to values
VIAME_CORE_EXPORT
std::map< std::string, double > evaluate_models(
  const std::vector< std::string >& computed_files,
  const std::vector< std::string >& groundtruth_files,
  const evaluation_config& config = evaluation_config() );

} // namespace viame

#endif // VIAME_CORE_EVALUATE_MODELS_H
