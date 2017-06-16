/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief core pos_io tests
 */

#include <test_common.h>

#include <iostream>
#include <sstream>

#include <vital/video_metadata/pos_metadata_io.h>
#include <vital/exceptions.h>


#define TEST_ARGS ( kwiver::vital::path_t const &data_dir )
DECLARE_TEST_MAP();


int main(int argc, char** argv)
{
  CHECK_ARGS(2);
  testname_t const testname = argv[1];
  kwiver::vital::path_t data_dir( argv[2] );
  RUN_TEST( testname, data_dir );
}


IMPLEMENT_TEST(pos_format_read)
{
  kwiver::vital::path_t test_read_file = data_dir + "/sample_pos.pos";
  auto input_md = kwiver::vital::read_pos_file( test_read_file );
  print_metadata(std::cout, *input_md);

  test_read_file = data_dir + "/sample_pos_no_name.pos";
  input_md = kwiver::vital::read_pos_file( test_read_file );
  print_metadata(std::cout, *input_md);
}


IMPLEMENT_TEST(invalid_file_path)
{
  EXPECT_EXCEPTION(
      kwiver::vital::file_not_found_exception,
      auto md = kwiver::vital::read_pos_file( data_dir + "/not_a_file.blob" ),
      "tried loading an invalid file path"
      );
}


IMPLEMENT_TEST(invalid_file_content)
{
  kwiver::vital::path_t invalid_content_file = data_dir + "/invalid_pos.pos";
  EXPECT_EXCEPTION(
      kwiver::vital::invalid_data,
      auto md = kwiver::vital::read_pos_file( invalid_content_file ),
      "tried loading a file with invalid data"
      );
}


IMPLEMENT_TEST(output_format_test)
{
  kwiver::vital::path_t test_read_file = data_dir + "/sample_pos.pos";
  auto input_md = kwiver::vital::read_pos_file( test_read_file );
  print_metadata(std::cout, *input_md);

  kwiver::vital::path_t temp_file = "temp.pos";
  kwiver::vital::write_pos_file( *input_md, temp_file );

  auto md = kwiver::vital::read_pos_file( temp_file );

  constexpr double epsilon = 1e-8;

  TEST_EQUAL("metadata has same size after IO", input_md->size(), md->size());

  for (auto mdi : *input_md)
  {
    if ( !md->has( mdi.second->tag() ) )
    {
      TEST_ERROR("Reloaded metadata is missing tag " << mdi.second->name());
      continue;
    }
    auto const& other_mdi = md->find( mdi.second->tag() );
    if ( mdi.second->type() == typeid(double) )
    {
      TEST_NEAR("Value of tag " << mdi.second->name(),
                mdi.second->as_double(), other_mdi.as_double(), epsilon);
    }
    else if ( mdi.second->type() == typeid(uint64_t) )
    {
      TEST_EQUAL("Value of tag " << mdi.second->name(),
                 mdi.second->as_uint64(), other_mdi.as_uint64());
    }
    else if ( mdi.second->type() == typeid(int) )
    {
      int v1=0, v2=0;
      mdi.second->data(v1);
      other_mdi.data(v2);
      TEST_EQUAL("Value of tag " << mdi.second->name(),
                 v1, v2);
    }
    else if ( mdi.second->type() == typeid(std::string) )
    {
      TEST_EQUAL("Value of tag " << mdi.second->name(),
                 mdi.second->as_string(), other_mdi.as_string());
    }
    else if ( mdi.second->type() == typeid(kwiver::vital::geo_lat_lon) )
    {
      kwiver::vital::geo_lat_lon v1, v2;
      mdi.second->data(v1);
      other_mdi.data(v2);
      TEST_NEAR("Value of tag " << mdi.second->name() << " (lat)",
                v1.latitude(), v2.latitude(), epsilon);
      TEST_NEAR("Value of tag " << mdi.second->name() << " (long)",
                v1.longitude(), v2.longitude(), epsilon);
    }
    else
    {
      std::cout << "Unable to compare tag " << mdi.second->name() << std::endl;
    }
  }
}
