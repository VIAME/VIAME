/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "evaluate_models.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Test fixture: writes VIAME CSV files to a scratch directory
// =============================================================================

class evaluate_models_test : public ::testing::Test
{
protected:
  void SetUp() override
  {
    m_dir = fs::temp_directory_path() /
            ( "viame_eval_test_" +
              std::to_string( reinterpret_cast< uintptr_t >( this ) ) );
    fs::create_directories( m_dir );
  }

  void TearDown() override
  {
    std::error_code ec;
    fs::remove_all( m_dir, ec );
  }

  /// One VIAME CSV row
  struct row
  {
    int track_id;
    int frame_id;
    double x1, y1, x2, y2;
    double confidence;
    std::string class_name;
  };

  /// Write rows to a VIAME CSV and return its path
  std::string write_csv( const std::string& name,
                         const std::vector< row >& rows )
  {
    const std::string path = ( m_dir / name ).string();

    std::ofstream file( path );
    file << "# 1: Detection or Track Id, 2: Video or Image String, "
         << "3: Frame Number, 4-7: Bounding Box, 8: Confidence, "
         << "9: Length, 10+: Class name / score pairs\n";

    for( const auto& r : rows )
    {
      file << r.track_id << ","
           << "frame_" << r.frame_id << ".png,"
           << r.frame_id << ","
           << r.x1 << "," << r.y1 << "," << r.x2 << "," << r.y2 << ","
           << r.confidence << ",0,"
           << r.class_name << "," << r.confidence << "\n";
    }

    return path;
  }

  /// A box that overlaps the given box exactly (IoU of 1)
  static row box( int track_id, int frame_id, double confidence,
                  const std::string& class_name = "fish",
                  double x = 10 )
  {
    return { track_id, frame_id, x, 10, x + 20, 30, confidence, class_name };
  }

  viame::evaluation_results evaluate(
    const std::vector< std::string >& computed,
    const std::vector< std::string >& truth,
    const viame::evaluation_config& config = viame::evaluation_config() )
  {
    viame::model_evaluator evaluator;
    evaluator.set_config( config );
    return evaluator.evaluate( computed, truth );
  }

  fs::path m_dir;
};

// =============================================================================
// Average precision
// =============================================================================

// A perfect detector scores an AP of one, and AP agrees with AP@50 at the
// default IoU threshold (both use confidence-ordered, all-point interpolation)
TEST_F( evaluate_models_test, ap_perfect_detector )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 2, 1, 1.0 ), box( 3, 2, 1.0 ) } );
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ), box( 2, 1, 0.8 ), box( 3, 2, 0.7 ) } );

  auto results = evaluate( { computed }, { truth } );

  EXPECT_DOUBLE_EQ( 3.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.false_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.false_negatives );
  EXPECT_NEAR( 1.0, results.average_precision, 1e-9 );
  EXPECT_NEAR( results.ap50, results.average_precision, 1e-9 );
}

// Hand-computed AP: two of three detections are true positives, and the
// false positive outranks the second true positive.
//
// Ordered by confidence: TP(0.9), FP(0.8), TP(0.7)
//   after 1: precision 1/1 = 1.000, recall 1/2 = 0.5
//   after 2: precision 1/2 = 0.500, recall 1/2 = 0.5
//   after 3: precision 2/3 = 0.667, recall 2/2 = 1.0
// Interpolating precision right to left gives [1.0, 0.667, 0.667], so
// AP = 1.0 * 0.5 + 0.667 * 0.0 + 0.667 * 0.5 = 0.8333...
TEST_F( evaluate_models_test, ap_hand_computed )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 2, 1, 1.0 ) } );
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ),                     // true positive
    { 2, 0, 500, 500, 520, 520, 0.8, "fish" },  // false positive, no overlap
    box( 3, 1, 0.7 ) } );                 // true positive

  auto results = evaluate( { computed }, { truth } );

  EXPECT_DOUBLE_EQ( 2.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 1.0, results.false_positives );
  EXPECT_NEAR( 0.5 + ( 2.0 / 3.0 ) * 0.5, results.average_precision, 1e-9 );
}

// AP@[0.5:0.95] must average all ten thresholds including 0.95. A float
// accumulation loop silently drops the last, strictest one.
TEST_F( evaluate_models_test, ap50_95_includes_strictest_threshold )
{
  // A box offset far enough that IoU is high but below 0.95
  auto truth = write_csv( "truth.csv", { { 1, 0, 0, 0, 100, 100, 1.0, "fish" } } );
  auto computed = write_csv( "computed.csv", { { 1, 0, 4, 0, 104, 100, 0.9, "fish" } } );

  auto results = evaluate( { computed }, { truth } );

  // IoU here is 96/104 = 0.923, so AP is one at every threshold up to 0.90 and
  // zero at 0.95. Averaging ten thresholds gives 0.9; averaging only the nine
  // up to 0.90 would wrongly give 1.0.
  EXPECT_NEAR( 1.0, results.ap50, 1e-9 );
  EXPECT_NEAR( 0.9, results.ap50_95, 1e-9 );
}

// =============================================================================
// Per-class metrics
// =============================================================================

// Per-class AP must be reported, and mean AP must be the mean of those values
// (not a mean of F1 scores, and not silently skipping zero-AP classes)
TEST_F( evaluate_models_test, per_class_ap_and_mean_ap )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0, "fish" ),
    box( 2, 1, 1.0, "fish" ),
    box( 3, 2, 1.0, "crab" ) } );

  // Every fish is found; the crab is missed entirely
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9, "fish" ),
    box( 2, 1, 0.8, "fish" ) } );

  viame::evaluation_config config;
  config.compute_per_class_metrics = true;

  auto results = evaluate( { computed }, { truth }, config );

  ASSERT_EQ( 1u, results.per_class_metrics.count( "fish" ) );
  ASSERT_EQ( 1u, results.per_class_metrics.count( "crab" ) );

  const auto& fish = results.per_class_metrics.at( "fish" );
  const auto& crab = results.per_class_metrics.at( "crab" );

  ASSERT_EQ( 1u, fish.count( "average_precision" ) );
  ASSERT_EQ( 1u, crab.count( "average_precision" ) );

  EXPECT_NEAR( 1.0, fish.at( "average_precision" ), 1e-9 );
  EXPECT_NEAR( 0.0, crab.at( "average_precision" ), 1e-9 );

  // The wholly missed class must drag the mean down, not be dropped from it
  EXPECT_NEAR( 0.5, results.mean_ap, 1e-9 );
}

// =============================================================================
// Ground truth handling
// =============================================================================

// The confidence threshold applies to computed detections only. Ground truth
// routinely carries a placeholder confidence, and filtering it would delete
// false negatives and inflate recall.
TEST_F( evaluate_models_test, confidence_threshold_never_filters_ground_truth )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 0.0 ), box( 2, 1, 0.0 ), box( 3, 2, 0.0 ) } );
  auto computed = write_csv( "computed.csv", { box( 1, 0, 0.9 ) } );

  viame::evaluation_config config;
  config.confidence_threshold = 0.5;

  auto results = evaluate( { computed }, { truth }, config );

  EXPECT_DOUBLE_EQ( 3.0, results.total_gt_objects );
  EXPECT_DOUBLE_EQ( 1.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 2.0, results.false_negatives );
  EXPECT_NEAR( 1.0 / 3.0, results.recall, 1e-9 );
}

// =============================================================================
// Multiple sequences
// =============================================================================

// Frame and track IDs repeat across sequences. Scoring two file pairs must not
// match one sequence's detections against another's ground truth.
TEST_F( evaluate_models_test, sequences_do_not_share_frame_ids )
{
  auto truth_a = write_csv( "truth_a.csv", { box( 1, 0, 1.0 ) } );
  auto computed_a = write_csv( "computed_a.csv", { box( 1, 0, 0.9 ) } );

  // Same frame numbers and track IDs, but boxes nowhere near sequence A's
  auto truth_b = write_csv( "truth_b.csv", {
    { 1, 0, 500, 500, 520, 520, 1.0, "fish" } } );
  auto computed_b = write_csv( "computed_b.csv", {
    { 1, 0, 500, 500, 520, 520, 0.9, "fish" } } );

  auto results = evaluate( { computed_a, computed_b }, { truth_a, truth_b } );

  // Each sequence matches within itself: two frames, two true positives, and
  // no cross-sequence false positives or negatives
  EXPECT_DOUBLE_EQ( 2.0, results.total_frames );
  EXPECT_DOUBLE_EQ( 2.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.false_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.false_negatives );
}

// =============================================================================
// Matthews correlation coefficient
// =============================================================================

// MCC is approximated from precision and recall, since true negatives are
// undefined for detection. A perfect detector scores one; substituting TN = 0
// into the textbook formula would instead yield a negative value here.
TEST_F( evaluate_models_test, mcc_is_positive_for_a_good_detector )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 2, 1, 1.0 ), box( 3, 2, 1.0 ) } );
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ), box( 2, 1, 0.8 ), box( 3, 2, 0.7 ) } );

  auto results = evaluate( { computed }, { truth } );

  EXPECT_NEAR( 1.0, results.mcc, 1e-9 );
}

TEST_F( evaluate_models_test, mcc_matches_precision_recall_approximation )
{
  auto truth = write_csv( "truth.csv", { box( 1, 0, 1.0 ), box( 2, 1, 1.0 ) } );
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ),                            // true positive
    { 2, 0, 500, 500, 520, 520, 0.8, "fish" } }  // false positive
  );

  auto results = evaluate( { computed }, { truth } );

  const double p = results.precision;  // 1 / 2
  const double r = results.recall;     // 1 / 2
  const double expected = std::sqrt( p * r ) -
                          std::sqrt( ( 1.0 - p ) * ( 1.0 - r ) );

  EXPECT_NEAR( expected, results.mcc, 1e-9 );
  EXPECT_NEAR( 0.0, results.mcc, 1e-9 );
}

// =============================================================================
// Plot generation
// =============================================================================

// A frame holding ground truth but no detections at all is absent from the
// computed frame index; generating a confusion matrix must tolerate that
// rather than throwing out of a map lookup.
TEST_F( evaluate_models_test, plots_survive_a_frame_with_no_detections )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 2, 1, 1.0 ), box( 3, 2, 1.0 ) } );

  // Nothing at all on frame 1
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ), box( 3, 2, 0.7 ) } );

  viame::model_evaluator evaluator;
  auto results = evaluator.evaluate( { computed }, { truth } );

  EXPECT_DOUBLE_EQ( 1.0, results.false_negatives );

  viame::evaluation_plot_data plot_data;
  ASSERT_NO_THROW( plot_data = evaluator.generate_plot_data() );

  EXPECT_FALSE( plot_data.confusion_matrix.class_names.empty() );
  EXPECT_FALSE( plot_data.overall_pr_curve.points.empty() );
}

// The detection ROC plots false alarms per frame, which is unbounded, so its
// summary statistic is the area normalized by that range and must stay in [0, 1]
TEST_F( evaluate_models_test, roc_curve_mean_pd_is_normalized )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 2, 1, 1.0 ) } );

  // Two hits plus four false alarms spread over two frames
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ),
    box( 2, 1, 0.8 ),
    { 3, 0, 200, 200, 220, 220, 0.7, "fish" },
    { 4, 0, 300, 300, 320, 320, 0.6, "fish" },
    { 5, 1, 400, 400, 420, 420, 0.5, "fish" },
    { 6, 1, 500, 500, 520, 520, 0.4, "fish" } } );

  viame::model_evaluator evaluator;
  evaluator.evaluate( { computed }, { truth } );

  const auto roc = evaluator.generate_roc_curve();

  ASSERT_FALSE( roc.points.empty() );

  // Four false alarms over two frames
  EXPECT_NEAR( 2.0, roc.max_false_alarms_per_frame, 1e-9 );

  // The curve is anchored at the origin
  EXPECT_NEAR( 0.0, roc.points.front().false_alarms_per_frame, 1e-9 );
  EXPECT_NEAR( 0.0, roc.points.front().true_positive_rate, 1e-9 );

  EXPECT_GE( roc.mean_pd, 0.0 );
  EXPECT_LE( roc.mean_pd, 1.0 );
}

// Track purity and continuity histograms are populated, not left as zeros
TEST_F( evaluate_models_test, track_histograms_are_populated )
{
  auto truth = write_csv( "truth.csv", {
    box( 1, 0, 1.0 ), box( 1, 1, 1.0 ), box( 1, 2, 1.0 ) } );
  auto computed = write_csv( "computed.csv", {
    box( 1, 0, 0.9 ), box( 1, 1, 0.9 ), box( 1, 2, 0.9 ) } );

  viame::model_evaluator evaluator;
  evaluator.evaluate( { computed }, { truth } );

  const auto plot_data = evaluator.generate_plot_data();

  auto total = []( const std::vector< int >& histogram )
  {
    int sum = 0;
    for( int count : histogram )
    {
      sum += count;
    }
    return sum;
  };

  // One computed track, perfectly pure and continuous, so it lands in the top
  // bin of both histograms
  EXPECT_EQ( 1, total( plot_data.track_purity_histogram ) );
  EXPECT_EQ( 1, total( plot_data.track_continuity_histogram ) );
  EXPECT_EQ( 1, plot_data.track_purity_histogram.back() );
  EXPECT_EQ( 1, plot_data.track_continuity_histogram.back() );
}

// =============================================================================
// Degenerate inputs
// =============================================================================

TEST_F( evaluate_models_test, empty_inputs_do_not_divide_by_zero )
{
  auto truth = write_csv( "truth.csv", {} );
  auto computed = write_csv( "computed.csv", {} );

  viame::model_evaluator evaluator;
  auto results = evaluator.evaluate( { computed }, { truth } );

  EXPECT_DOUBLE_EQ( 0.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.precision );
  EXPECT_DOUBLE_EQ( 0.0, results.recall );
  EXPECT_TRUE( std::isfinite( results.average_precision ) );
  EXPECT_TRUE( std::isfinite( results.mcc ) );

  ASSERT_NO_THROW( evaluator.generate_plot_data() );
}

TEST_F( evaluate_models_test, detections_with_no_ground_truth_are_all_false_positives )
{
  auto truth = write_csv( "truth.csv", {} );
  auto computed = write_csv( "computed.csv", { box( 1, 0, 0.9 ), box( 2, 1, 0.8 ) } );

  auto results = evaluate( { computed }, { truth } );

  EXPECT_DOUBLE_EQ( 0.0, results.true_positives );
  EXPECT_DOUBLE_EQ( 2.0, results.false_positives );
  EXPECT_DOUBLE_EQ( 0.0, results.average_precision );
  EXPECT_TRUE( std::isfinite( results.mcc ) );
}
