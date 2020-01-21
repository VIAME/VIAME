/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
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
#include <arrows/tests/test_image.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
namespace gdal = kwiver::arrows::gdal;
static int expected_size = 32;
static std::string geotiff_file_name = "test.tif";
static std::string nitf_file_name = "test.ntf";
static std::string jpeg_file_name = "test.jpg";
static std::string png_file_name = "test.png";
static std::vector<int> test_x_pixels = {0, 3, 11, 21, 31};
static std::vector<int> test_y_pixels = {0, 5, 8, 13, 31};

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

static std::vector< kwiver::vital::vital_metadata_tag > nitf_tags =
{
  kwiver::vital::VITAL_META_NITF_IDATIM,
  kwiver::vital::VITAL_META_NITF_BLOCKA_FRFC_LOC_01,
  kwiver::vital::VITAL_META_NITF_BLOCKA_FRLC_LOC_01,
  kwiver::vital::VITAL_META_NITF_BLOCKA_LRLC_LOC_01,
  kwiver::vital::VITAL_META_NITF_BLOCKA_LRFC_LOC_01,
  kwiver::vital::VITAL_META_NITF_IMAGE_COMMENTS,
};

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
void test_rpc_metadata(kwiver::vital::metadata_sptr md)
{
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

// ----------------------------------------------------------------------------
void test_nitf_metadata(kwiver::vital::metadata_sptr md)
{

  kwiver::vital::metadata_traits md_traits;
  for ( auto const& tag : nitf_tags )
  {
    EXPECT_TRUE( md->has( tag ) )
      << "Image metadata should include " << md_traits.tag_to_name( tag );
  }

  if (md->size() > 0)
  {
    kwiver::vital::print_metadata( std::cout, *md );
  }
}

// ----------------------------------------------------------------------------
class image_io : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(image_io, create)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  ASSERT_NE(nullptr, algo::image_io::create("gdal"));
}

TEST_F(image_io, load_geotiff)
{
  kwiver::arrows::gdal::image_io img_io;

  kwiver::vital::path_t file_path = data_dir + "/" + geotiff_file_name;
  auto img_ptr = img_io.load(file_path);

  EXPECT_EQ( img_ptr->width(), expected_size );
  EXPECT_EQ( img_ptr->height(), expected_size );
  EXPECT_EQ( img_ptr->depth(), 1 );

  // Test some pixel values
  kwiver::vital::image_of<uint16_t> img(img_ptr->get_image());
  for ( auto x_px : test_x_pixels )
  {
    for ( auto y_px : test_y_pixels )
    {
      uint16_t expected_pixel_value = (std::numeric_limits<uint16_t>::max() + 1)
        *x_px*y_px/expected_size/expected_size;
      EXPECT_EQ( img(x_px, y_px), expected_pixel_value);
    }
  }

  auto md = img_ptr->get_metadata();

  test_rpc_metadata(md);

  // Test corner points
  ASSERT_TRUE( md->has( kwiver::vital::VITAL_META_CORNER_POINTS ) )
    << "Metadata should include corner points.";

  kwiver::vital::geo_polygon corner_pts;
  md->find( kwiver::vital::VITAL_META_CORNER_POINTS ).data( corner_pts );
  EXPECT_EQ( corner_pts.crs(), 4326);
  EXPECT_TRUE( corner_pts.polygon( 4326 ).contains( -16.0, 0.0) );
  EXPECT_TRUE( corner_pts.polygon( 4326 ).contains( 0.0, 32.0) );
  EXPECT_TRUE( corner_pts.polygon( 4326 ).contains( 0.0, -32.0) );
  EXPECT_TRUE( corner_pts.polygon( 4326 ).contains( 16.0, 0.0) );
}

TEST_F(image_io, load_nitf)
{
  kwiver::arrows::gdal::image_io img_io;

  kwiver::vital::path_t file_path = data_dir + "/" + nitf_file_name;
  auto img_ptr = img_io.load(file_path);

  EXPECT_EQ( img_ptr->width(), expected_size );
  EXPECT_EQ( img_ptr->height(), expected_size );
  EXPECT_EQ( img_ptr->depth(), 1 );

  // Test some pixel values
  kwiver::vital::image_of<float> img(img_ptr->get_image());
  for ( auto x_px : test_x_pixels )
  {
    for ( auto y_px : test_y_pixels )
    {
      float expected_pixel_value = x_px*y_px/float(expected_size*expected_size);
      EXPECT_EQ( img(x_px, y_px), expected_pixel_value);
    }
  }

  auto md = img_ptr->get_metadata();

  test_rpc_metadata(md);
}

TEST_F(image_io, load_nitf_2)
{
  kwiver::arrows::gdal::image_io img_io;
  kwiver::vital::path_t file_path = data_dir + "/" + nitf_file_name;
  auto img_ptr = img_io.load(file_path);

  EXPECT_EQ( img_ptr->width(), 32);
  EXPECT_EQ( img_ptr->height(), 32);
  EXPECT_EQ( img_ptr->depth(), 1 );

  auto md = img_ptr->get_metadata();
  test_nitf_metadata(md);
}

TEST_F(image_io, load_jpeg)
{
  kwiver::arrows::gdal::image_io img_io;

  kwiver::vital::path_t file_path = data_dir + "/" + jpeg_file_name;
  auto img_ptr = img_io.load(file_path);

  EXPECT_EQ( img_ptr->width(), expected_size );
  EXPECT_EQ( img_ptr->height(), expected_size );
  EXPECT_EQ( img_ptr->depth(), 3 );

  uint8_t norm_fact =
    expected_size*expected_size/(std::numeric_limits<uint8_t>::max() + 1);

  // Test some pixel values
  kwiver::vital::image_of<uint8_t> img(img_ptr->get_image());
  for ( auto x_px : test_x_pixels )
  {
    for ( auto y_px : test_y_pixels )
    {
      auto pixel = img.at( x_px, y_px );

      uint8_t expected_red = x_px*y_px/norm_fact;
      uint8_t expected_blue = (expected_size - x_px - 1)*y_px/norm_fact;
      uint8_t expected_green = x_px*(expected_size - y_px - 1)/norm_fact;
      // Due to lossy compression exact comparisons will fail
      EXPECT_NEAR( pixel.r, expected_red, 1 )
        << "Incorrect red value at pixel (" << x_px << "," << y_px << ")";
      EXPECT_NEAR( pixel.b, expected_blue,1 )
        << "Incorrect blue value at pixel (" << x_px << "," << y_px << ")";
      EXPECT_NEAR( pixel.g, expected_green, 1 )
        << "Incorrect green value at pixel (" << x_px << "," << y_px << ")";
    }
  }
}

// ----------------------------------------------------------------------------
class get_image : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(get_image, crop)
{
  kwiver::arrows::gdal::image_io img_io;

  kwiver::vital::path_t file_path = data_dir + "/" + png_file_name;

  auto img_cont = img_io.load(file_path);

  test_get_image_crop<uint8_t>( img_cont );
}

