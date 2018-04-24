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

#include <arrows/tests/test_image.h>

#include <arrows/gdal/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(image, create)
{
  plugin_manager::instance().load_all_plugins();

  std::shared_ptr<algo::image_io> img_io;
  ASSERT_NE(nullptr, img_io = algo::image_io::create("gdal"));

  algo::image_io* img_io_ptr = img_io.get();
  EXPECT_EQ(typeid(gdal::image_io), typeid(*img_io_ptr))
    << "Factory method did not construct the correct type";
}

TEST(image, open)
{
  plugin_manager::instance().load_all_plugins();

  auto img_io = algo::image_io::create("gdal");

  // TODO: Hard code for now until we settle on a test file
  std::string filepath =
    "/Users/chet.nieter/projects/ComputerVision/Core3D/data_fouo/"
    "16JAN26183049-M1BS-500647790060_01_P001.tif";

  auto img_ptr = img_io->load(filepath);

  EXPECT_EQ( img_ptr->width(), 300 );
  EXPECT_EQ( img_ptr->height(), 300 );
  EXPECT_EQ( img_ptr->depth(), 2 );

  auto md = img_ptr->get_metadata();

  if (md->size() > 0)
  {
    std::cout << "-----------------------------------\n" << std::endl;
    kwiver::vital::print_metadata( std::cout, *md );
  }
}

