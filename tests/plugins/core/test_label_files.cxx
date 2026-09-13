/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>
#include "utilities_file.h"
#include "utilities_training.h"
#include <chrono>
#include <filesystem>
#include <fstream>

class label_files : public ::testing::Test
{
protected:
  std::filesystem::path directory;
  void SetUp() override
  {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
      ( "viame-label-files-" + std::to_string( stamp ) );
    ASSERT_TRUE( std::filesystem::create_directory( directory ) );
  }
  void TearDown() override { std::filesystem::remove_all( directory ); }
  std::string write( const std::string& extension, const std::string& content )
  {
    const auto path = ( directory / ( "labels" + extension ) ).string();
    std::ofstream out( path, std::ios::binary );
    out << content;
    return path;
  }
};

TEST_F( label_files, discovery_prefers_txt_then_csv_then_json )
{
  EXPECT_TRUE( viame::find_labels_file( directory.string() ).empty() );
  const auto json = write( ".json", "[]" );
  EXPECT_EQ( viame::find_labels_file( directory.string() ), json );
  const auto csv = write( ".csv", "fish" );
  EXPECT_EQ( viame::find_labels_file( directory.string() ), csv );
  const auto txt = write( ".txt", "fish" );
  EXPECT_EQ( viame::find_labels_file( directory.string() ), txt );
}

TEST_F( label_files, annotation_discovery_identifies_label_files )
{
  for( const auto& extension : { ".txt", ".csv", ".json" } )
  {
    EXPECT_TRUE( viame::is_labels_file( write( extension, "" ) ) );
  }
  const auto selected = ( directory / "categories.csv" ).string();
  EXPECT_TRUE( viame::is_labels_file(
    ( directory / "." / "categories.csv" ).string(), selected ) );
  EXPECT_FALSE( viame::is_labels_file(
    ( directory / "groundtruth.csv" ).string(), selected ) );
}

TEST_F( label_files, training_maps_aliases_and_retains_hierarchy_in_all_formats )
{
  const std::map< std::string, std::string > formats = {
    { ".txt", "\"sport glove\" \"athletic glove\" :parent=\"sport equipment\"\n"
                "glove\n\"sport equipment\"\n" },
    { ".csv", "sport glove,athletic glove,:parent=sport equipment\nglove\nsport equipment\n" },
    { ".json", R"({"categories":[
        {"name":"sport glove","synonyms":["athletic glove"],
         "supercategory":"sport equipment"},"glove","sport equipment"]})" }
  };
  for( const auto& format : formats )
  {
    SCOPED_TRACE( format.first );
    auto labels = std::make_shared< kwiver::vital::category_hierarchy >(
      write( format.first, format.second ) );
    auto truth = std::make_shared< kwiver::vital::detected_object_set >();
    for( const auto& name : { "athletic glove", "glove", "unlisted" } )
    {
      truth->add( std::make_shared< kwiver::vital::detected_object >( 1.0,
        std::make_shared< kwiver::vital::detected_object_type >( name, 0.9 ) ) );
    }
    EXPECT_TRUE( viame::adjust_labels( truth, labels, {} ) );
    ASSERT_EQ( truth->size(), 2 );
    std::string name;
    truth->at( 0 )->type()->get_most_likely( name );
    EXPECT_EQ( name, "sport glove" );
    truth->at( 1 )->type()->get_most_likely( name );
    EXPECT_EQ( name, "glove" );
    EXPECT_EQ( labels->get_class_parents( "sport glove" ),
      ( std::vector< std::string >{ "sport equipment" } ) );
  }
}
