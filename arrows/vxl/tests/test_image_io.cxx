// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL image class functionality
 */

#include <test_gtest.h>
#include <test_tmpfn.h>

#include <arrows/tests/test_image.h>

#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <kwiversys/SystemTools.hxx>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/transform_image.h>

#include <gtest/gtest.h>

namespace kv = kwiver::vital;
namespace ka = kwiver::arrows;
using ST = kwiversys::SystemTools;

kv::path_t g_data_dir;
static std::string test_color_image_name = "images/small_color_logo.png";
static std::string test_plane_image_name = "images/planes_logo.png";

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();

  GET_ARG( 1, g_data_dir );

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class image_io : public ::testing::Test
{
  TEST_ARG( data_dir );
};

// ----------------------------------------------------------------------------
TEST_F(image_io, save_plane)
{
  // Create an image to output
  auto const& vil_image = vil_image_view< vxl_byte >( 150, 150, 3 );
  auto image = ka::vxl::image_container( vil_image );
  auto image_ptr = std::make_shared< ka::vxl::image_container >( image );

  // Configure to split channels
  ka::vxl::image_io io;
  auto config = kv::config_block::empty_config();
  config->set_value( "split_channels", true );
  io.set_configuration( config );

  auto const& output_filename =
    kwiver::testing::temp_file_name( "image_io_save_plane-", ".png" );

  io.save( output_filename, image_ptr );

  auto const reread_image_ptr = io.load( output_filename );

  EXPECT_TRUE( equal_content( image_ptr->get_image(),
                              reread_image_ptr->get_image() ) );

  const auto saved_filenames = io.plane_filenames( output_filename );

  for( auto const& saved_filename : saved_filenames )
  {
    if( !ST::RemoveFile( saved_filename ) )
    {
      std::cerr << "Failed to remove output vxl plane image" << std::endl;
    }
  }
}

// ----------------------------------------------------------------------------
TEST_F(image_io, load_plane)
{
  std::string color_filename = data_dir + "/" + test_color_image_name;
  std::string plane_filename = data_dir + "/" + test_plane_image_name;

  ka::vxl::image_io reader;
  auto const color_image_ptr = reader.load( color_filename );

  auto config = kv::config_block::empty_config();
  config->set_value( "split_channels", true );
  reader.set_configuration( config );

  auto const plane_image_ptr = reader.load( plane_filename );
  EXPECT_TRUE( equal_content( color_image_ptr->get_image(),
                              plane_image_ptr->get_image() ) );
}
