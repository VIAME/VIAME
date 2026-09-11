/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What `viame score --output-plots` writes into its directory
///
/// The plots themselves were drawn by `plugins/opencv/plot_metrics`, 1115
/// lines of OpenCV primitives, and P7-T07 replaces them with matplotlib in
/// `tools/plot.py`. That is a rewrite rather than a port -- the picture at
/// the end is a different picture on purpose -- so there is no pixel
/// recording to hold it to. What can be recorded, and what actually has to
/// survive, is the **numbers the plots are drawn from**: the CSV files
/// `export_plot_data` writes beside them. A renderer reading those files can
/// only draw what they contain, so this is the contract between the two.
///
/// The expected output is committed beside this file; regenerate it with
///
///     VIAME_RECORD_PLOT_DATA_EXPORT=1 ./tests/bin/test-viame_core-plot_data_export
///
/// which writes the file and fails, so a regeneration is never accidental.
///
/// The scenario is built to reach every field of `evaluation_plot_data` at
/// once: two classes so there are per-class curves and an off-diagonal
/// confusion cell, tracks spanning several frames so the length, purity and
/// continuity histograms are populated, detections that miss their truth by
/// a known margin so the IoU histogram spreads over more than one bin, and
/// false alarms on their own frames so the ROC curve has somewhere to go.

#include <gtest/gtest.h>

#include "evaluate_models.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ----------------------------------------------------------------------------
std::string
expected_path()
{
#ifdef VIAME_PLOT_DATA_EXPORT_DIR
  return std::string( VIAME_PLOT_DATA_EXPORT_DIR ) + "/plot_data_export.txt";
#else
  return "plot_data_export.txt";
#endif
}

// ----------------------------------------------------------------------------
struct row
{
  int track_id;
  int frame_id;
  double x1, y1, x2, y2;
  double confidence;
  std::string class_name;
};

// ----------------------------------------------------------------------------
std::string
write_csv( const fs::path& dir, const std::string& name,
           const std::vector< row >& rows )
{
  const std::string path = ( dir / name ).string();

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

// ----------------------------------------------------------------------------
/// Every file the export wrote, in name order, as one comparable document.
std::string
read_back( const fs::path& dir )
{
  std::vector< fs::path > files;

  for( const auto& entry : fs::directory_iterator( dir ) )
  {
    if( entry.is_regular_file() )
    {
      files.push_back( entry.path() );
    }
  }

  std::sort( files.begin(), files.end() );

  std::ostringstream out;

  for( const auto& file : files )
  {
    out << "=== " << file.filename().string() << " ===\n";

    std::ifstream stream( file );
    out << stream.rdbuf();

    out << "\n";
  }

  return out.str();
}

} // namespace

// ----------------------------------------------------------------------------
class plot_data_export_test : public ::testing::Test
{
protected:
  void SetUp() override
  {
    m_dir = fs::temp_directory_path() /
            ( "viame_plot_export_" +
              std::to_string( reinterpret_cast< uintptr_t >( this ) ) );
    fs::create_directories( m_dir );
  }

  void TearDown() override
  {
    std::error_code ec;
    fs::remove_all( m_dir, ec );
  }

  fs::path m_dir;
};

// ----------------------------------------------------------------------------
TEST_F( plot_data_export_test, the_plot_directory_is_what_it_was )
{
  // Two classes over six frames. Track 1 is a clean `fish` track; track 2 is
  // a `scallop` track the detector loses in the middle, which is what puts
  // it below the top continuity bin; track 3 is truth the detector calls a
  // fish, which is the off-diagonal confusion cell.
  const std::vector< row > truth = {
    { 1, 0, 10, 10, 30, 30, 1.0, "fish" },
    { 1, 1, 12, 10, 32, 30, 1.0, "fish" },
    { 1, 2, 14, 10, 34, 30, 1.0, "fish" },
    { 1, 3, 16, 10, 36, 30, 1.0, "fish" },

    { 2, 0, 100, 100, 120, 120, 1.0, "scallop" },
    { 2, 1, 102, 100, 122, 120, 1.0, "scallop" },
    { 2, 2, 104, 100, 124, 120, 1.0, "scallop" },
    { 2, 3, 106, 100, 126, 120, 1.0, "scallop" },
    { 2, 4, 108, 100, 128, 120, 1.0, "scallop" },
    { 2, 5, 110, 100, 130, 120, 1.0, "scallop" },

    { 3, 4, 200, 200, 220, 220, 1.0, "scallop" },
  };

  // Track 1 is offset by a pixel or two, which spreads the IoU histogram;
  // track 2 is missing on frames 2 and 3; track 4 is pure false alarm.
  const std::vector< row > computed = {
    { 1, 0, 10, 10, 30, 30, 0.95, "fish" },
    { 1, 1, 13, 10, 33, 30, 0.90, "fish" },
    { 1, 2, 16, 10, 36, 30, 0.85, "fish" },
    { 1, 3, 20, 10, 40, 30, 0.55, "fish" },

    { 2, 0, 100, 100, 120, 120, 0.80, "scallop" },
    { 2, 1, 103, 100, 123, 120, 0.75, "scallop" },
    { 2, 4, 109, 100, 129, 120, 0.70, "scallop" },
    { 2, 5, 112, 100, 132, 120, 0.65, "scallop" },

    { 3, 4, 200, 200, 220, 220, 0.60, "fish" },

    { 4, 2, 300, 300, 320, 320, 0.50, "fish" },
    { 4, 3, 302, 300, 322, 320, 0.45, "scallop" },
  };

  const auto truth_path = write_csv( m_dir, "truth.csv", truth );
  const auto computed_path = write_csv( m_dir, "computed.csv", computed );

  viame::model_evaluator evaluator;
  evaluator.evaluate( { computed_path }, { truth_path } );

  const auto plot_data = evaluator.generate_plot_data();

  const auto out_dir = m_dir / "plots";
  fs::create_directories( out_dir );

  ASSERT_TRUE( viame::model_evaluator::export_plot_data(
                 plot_data, out_dir.string() ) );

  const std::string actual = read_back( out_dir );

  if( std::getenv( "VIAME_RECORD_PLOT_DATA_EXPORT" ) )
  {
    std::ofstream file( expected_path() );
    ASSERT_TRUE( file.is_open() ) << "cannot write " << expected_path();
    file << actual;
    file.close();

    FAIL() << "Recorded " << expected_path() << ". Unset "
              "VIAME_RECORD_PLOT_DATA_EXPORT and run again.";
  }

  std::ifstream expected_file( expected_path() );
  ASSERT_TRUE( expected_file.is_open() )
    << "no recording at " << expected_path() << "; regenerate with "
       "VIAME_RECORD_PLOT_DATA_EXPORT=1";

  std::ostringstream expected;
  expected << expected_file.rdbuf();

  EXPECT_EQ( expected.str(), actual );
}
