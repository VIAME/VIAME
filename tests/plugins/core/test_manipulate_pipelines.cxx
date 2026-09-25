/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "manipulate_pipelines.h"
#include "utilities_file.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

TEST( manipulate_pipelines, relative_paths_have_one_separator )
{
  for( const std::string detector : { "rf_detr", "yolo" } )
  {
    const std::string filename = "model files/checkpoint.pth";
    const std::string output = viame::format_output_as_pipe_blocks(
      { { "type", detector }, { detector + ":weight", filename } },
      { filename }, "  " );
    EXPECT_NE( output.find( "relativepath weight = " + filename + "\n" ),
               std::string::npos );
  }
}

TEST( manipulate_pipelines, template_outputs_use_lf )
{
  const std::string path = testing::TempDir() + "viame_template_endings.pipe";
  struct cleanup
  {
    std::string path;
    ~cleanup() { std::remove( path.c_str() ); }
  } remove_file{ path };

  // Check LF, Windows CRLF, legacy CR, and mixed template line endings.
  for( const std::string ending : { "\n", "\r\n", "\r", "\r\n\n" } )
  {
    {
      std::ofstream file( path, std::ios::binary );
      file << "process detector" << ending << "  [-IMPL-]" << ending;
      ASSERT_TRUE( file.good() );
    }
    const std::map< std::string, std::string > replacements = {
      { "[-IMPL-]", "block detector\r\n    relativepath weight = model.pth\rendblock" }
    };
    const std::string normalized_ending = ending == "\r\n\n" ? "\n\n" : "\n";
    const std::string expected = "process detector" + normalized_ending +
      "  block detector\n    relativepath weight = model.pth\nendblock" +
      normalized_ending;

    std::string content;
    ASSERT_TRUE( viame::replace_keywords_in_template_to_string(
      path, replacements, content ) );
    EXPECT_EQ( content, expected );

    // Training can render back into an existing pipeline in a later stage.
    ASSERT_TRUE( viame::replace_keywords_in_template_file(
      path, path, replacements ) );
    std::ifstream file( path, std::ios::binary );
    const std::string written( ( std::istreambuf_iterator< char >( file ) ),
                              std::istreambuf_iterator< char >() );
    EXPECT_EQ( written, expected );
  }
}

TEST( manipulate_pipelines, tracker_blocks_use_track_objects_root )
{
  const std::string output = viame::format_output_as_pipe_blocks(
    { { "type", "bytetrack" },
      { "bytetrack:high_thresh", "0.500" },
      { "bytetrack:track_buffer", "30" } },
    {}, "  ", "track_objects" );

  EXPECT_EQ( output,
    ":track_objects:type                          bytetrack\n"
    "  \n"
    "  block track_objects:bytetrack\n"
    "    :high_thresh                               0.500\n"
    "    :track_buffer                              30\n"
    "  endblock" );
}

TEST( manipulate_pipelines, tracker_impl_skips_special_keys )
{
  const std::string dir = testing::TempDir();
  const std::string template_path = dir + "viame_tracker_impl.pipe";
  const std::string params_path = dir + "viame_botsort_params.json";
  struct cleanup
  {
    std::vector< std::string > paths;
    ~cleanup() { for( const auto& p : paths ) { std::remove( p.c_str() ); } }
  } remove_files{ { template_path, params_path } };

  {
    std::ofstream file( template_path, std::ios::binary );
    file << "process detector\n  [-DETECTOR-IMPL-]\n\n"
            "process tracker\n  :: track_objects\n  [-TRACKER-IMPL-]\n";
    std::ofstream params( params_path, std::ios::binary );
    params << "{}";
    ASSERT_TRUE( file.good() && params.good() );
  }

  const std::map< std::string, std::string > output_map = {
    { "type", "botsort" },
    { "botsort:params_file", "botsort_params.json" },
    { "botsort_params.json", params_path },
    { "eval_folder", dir },
    { "tracker_pipeline_template", "templates/embedded_tracker.pipe" } };

  const std::string impl =
    viame::generate_tracker_impl_replacement( output_map, template_path );

  EXPECT_EQ( impl,
    ":track_objects:type                          botsort\n"
    "  \n"
    "  block track_objects:botsort\n"
    "    relativepath params_file = botsort_params.json\n"
    "  endblock" );

  // The detector pass must leave the tracker slot for the tracker pass.
  std::string rendered;
  ASSERT_TRUE( viame::replace_keywords_in_template_to_string( template_path,
    { { "[-DETECTOR-IMPL-]", "block detector\n  endblock" } }, rendered ) );
  EXPECT_NE( rendered.find( "  [-TRACKER-IMPL-]\n" ), std::string::npos );
}
