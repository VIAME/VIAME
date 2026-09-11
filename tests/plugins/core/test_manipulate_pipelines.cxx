/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "manipulate_pipelines.h"
#include "utilities_file.h"

#include <cstdio>
#include <fstream>
#include <iterator>

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
