/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "utilities_model_card.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace kv = kwiver::vital;

class utilities_model_card : public ::testing::Test
{
protected:
  std::filesystem::path directory;
  viame::model_card_inputs inputs;

  void SetUp() override
  {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
      ( "viame-model-card-" + std::to_string( stamp ) );
    inputs.output_directory = ( directory / "trained_model" ).string();
    ASSERT_TRUE( std::filesystem::create_directories( inputs.output_directory ) );
  }

  void TearDown() override { std::filesystem::remove_all( directory ); }

  std::string read( const std::string& name )
  {
    std::ifstream file( std::filesystem::path( inputs.output_directory ) / name );
    EXPECT_TRUE( file.is_open() );
    return { std::istreambuf_iterator< char >( file ), std::istreambuf_iterator< char >() };
  }

  kv::detected_object_set_sptr truth( const std::vector< std::string >& names )
  {
    auto result = std::make_shared< kv::detected_object_set >();
    for( const auto& name : names )
    {
      auto type = name.empty() ? nullptr
        : std::make_shared< kv::detected_object_type >( name, 1.0 );
      result->add( std::make_shared< kv::detected_object >( 1.0, type ) );
    }
    return result;
  }

  void contains( const std::string& text, const std::string& expected )
  {
    EXPECT_NE( text.find( expected ), std::string::npos ) << expected;
  }
};

TEST_F( utilities_model_card, detector_counts_and_explicit_splits )
{
  inputs.detector_types = { "wrapper (detector)" };
  inputs.detector_pipeline = "pipelines/detector.pipe";
  inputs.config_file = "configs/train.conf";
  inputs.init_weights = "models/seed.pth";
  inputs.items = { "train-sequence", "validation-sequence" };
  inputs.item_frame_counts = { 3, 1 };
  inputs.validation_items = { 1 };
  inputs.train_frames = { "frame-a.png", "frame-b.png", "frame-c.png" };
  inputs.train_truth = { truth( { "fish", "fish", "" } ), truth( {} ), nullptr };
  inputs.validation_frames = { "frame-d.png" };
  inputs.validation_truth = { truth( { "crab" } ) };

  viame::write_model_card( inputs );

  const auto card = read( "MODEL_CARD.md" );
  contains( card, "pipeline_tag: object-detection" );
  contains( card, "- detector\n- wrapper\n" );
  contains( card, "# Trained VIAME model" );
  contains( card, "| Detector pipeline | `detector.pipe` |" );
  contains( card, "| Training config | `train.conf` |" );
  contains( card, "| Seed weights | `seed.pth` |" );
  contains( card, "| fish |  | 2 | 0 |" );
  contains( card, "| unlabeled |  | 1 | 0 |" );
  contains( card, "| crab |  | 0 | 1 |" );
  contains( card, "| **Total** | | 3 | 1 |" );
  contains( card, "| Train | 1 | 3 | 1 | 3 |" );
  contains( card, "| Validation | 1 | 1 | 1 | 1 |" );
  contains( card, "### Validation\n\n- `validation-sequence` (1 frames)" );
  contains( card, "No held-out test set was scored." );
  EXPECT_EQ( read( "splits/train_frames.txt" ), "frame-a.png\nframe-b.png\nframe-c.png\n" );
  EXPECT_EQ( read( "splits/validation_frames.txt" ), "frame-d.png\n" );
}

TEST_F( utilities_model_card, declared_categories_synonyms_and_markdown_escaping )
{
  inputs.labels = std::make_shared< kv::category_hierarchy >();
  inputs.labels->add_class( "fish|skate" );
  inputs.labels->add_synonym( "fish|skate", "ray|fish" );
  inputs.labels->add_class( "unused" );
  inputs.train_truth = { truth( { "fish|skate" } ) };

  viame::write_model_card( inputs );

  const auto card = read( "MODEL_CARD.md" );
  contains( card, "2 categories, as declared by the labels file" );
  contains( card, "| fish\\|skate | ray\\|fish | 1 | 0 |" );
  contains( card, "| unused |  | 0 | 0 |" );
}

TEST_F( utilities_model_card, automatic_validation_and_evaluation_summary )
{
  inputs.detector_types = { "detector" };
  inputs.items = { "train-sequence" };
  inputs.validation_auto_selected = true;
  inputs.validation_percent = 0.05;
  inputs.validation_frames = { "held-out.png" };
  inputs.test_items = { "test-sequence" };
  inputs.test_results_dir = inputs.output_directory + "/test_results";
  std::filesystem::create_directory( inputs.test_results_dir );
  std::ofstream( inputs.test_results_dir + "/test_summary.txt" ) << "mAP: 0.75\n";

  viame::write_model_card( inputs );

  const auto card = read( "MODEL_CARD.md" );
  contains( card, "| Validation | (from train) | 1 | 0 | 0 |" );
  contains( card, "held out 1 frame (target 5% of the training frames, in bursts)" );
  contains( card, "### Test\n\n- `test-sequence`" );
  contains( card, "`test_results/`" );
  contains( card, "```\nmAP: 0.75\n```" );
  EXPECT_EQ( read( "splits/validation_frames.txt" ), "held-out.png\n" );
}

TEST_F( utilities_model_card, tracker_without_validation_or_test_scores )
{
  inputs.tracker_types = { "tracker" };
  inputs.tracker_pipeline = "tracker.pipe";
  inputs.gt_frames_only = true;
  inputs.test_items = { "unscored-sequence" };

  viame::write_model_card( inputs );

  const auto card = read( "MODEL_CARD.md" );
  contains( card, "pipeline_tag: object-tracking" );
  contains( card, "| Tracker pipeline | `tracker.pipe` |" );
  contains( card, "| Frames used | annotated frames only |" );
  contains( card, "No validation set was used." );
  contains( card, "A test set was given but could not be scored" );
  EXPECT_EQ( read( "splits/train_frames.txt" ), "" );
  EXPECT_EQ( read( "splits/validation_frames.txt" ), "" );
}
