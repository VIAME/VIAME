/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief test GDAL image class
 */

#include <test_gtest.h>

#include <arrows/gdal/image_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/metadata_traits.h>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
namespace gdal = kwiver::arrows::gdal;
static std::string nitf_file_name = "test.tif";

static std::vector< kwiver::vital::vital_metadata_tag > rpc_tags =
{
  kwiver::vital::VITAL_META_RPC_HEIGHT_OFFSET,
  kwiver::vital::VITAL_META_RPC_HEIGHT_SCALE,
  kwiver::vital::VITAL_META_RPC_LONG_OFFSET,
  kwiver::vital::VITAL_META_RPC_LONG_SCALE,
  kwiver::vital::VITAL_META_RPC_LAT_OFFSET,
  kwiver::vital::VITAL_META_RPC_LAT_SCALE,
  kwiver::vital::VITAL_META_RPC_ROW_OFFSET,
  kwiver::vital::VITAL_META_RPC_ROW_SCALE,
  kwiver::vital::VITAL_META_RPC_COL_OFFSET,
  kwiver::vital::VITAL_META_RPC_COL_SCALE ,
  kwiver::vital::VITAL_META_RPC_ROW_NUM_COEFF,
  kwiver::vital::VITAL_META_RPC_ROW_DEN_COEFF,
  kwiver::vital::VITAL_META_RPC_COL_NUM_COEFF,
  kwiver::vital::VITAL_META_RPC_COL_DEN_COEFF,
};

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class image_io : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(image_io, create)
{
  std::shared_ptr<algo::image_io> img_io;
  ASSERT_NE(nullptr, img_io = algo::image_io::create("gdal"));

  algo::image_io* img_io_ptr = img_io.get();
  EXPECT_EQ(typeid(gdal::image_io), typeid(*img_io_ptr))
    << "Factory method did not construct the correct type";
}

TEST_F(image_io, load)
{
  auto img_io = algo::image_io::create("gdal");

  kwiver::vital::path_t file_path = data_dir + "/" + nitf_file_name;
  auto img_ptr = img_io->load(file_path);

  EXPECT_EQ( img_ptr->width(), 50 );
  EXPECT_EQ( img_ptr->height(), 50 );
  EXPECT_EQ( img_ptr->depth(), 3 );

  auto md = img_ptr->get_metadata();

  EXPECT_EQ( md->size(), 14 )
    << "Image metadata should have 14 entries";

  kwiver::vital::metadata_traits md_traits;
  for ( auto const& tag : rpc_tags )
  {
    EXPECT_TRUE( md->has( tag ) )
      << "Image metadata should include " << md_traits.tag_to_name( tag );
  }

  if (md->size() > 0)
  {
    std::cout << "-----------------------------------\n" << std::endl;
    kwiver::vital::print_metadata( std::cout, *md );
  }
}

