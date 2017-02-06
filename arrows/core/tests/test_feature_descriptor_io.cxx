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

#include <test_common.h>

#include <arrows/core/feature_descriptor_io.h>
#include <arrows/core/register_algorithms.h>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  kwiver::arrows::core::register_algorithms();

  RUN_TEST(testname);
}

using namespace kwiver::vital;


IMPLEMENT_TEST(create)
{
  algo::feature_descriptor_io_sptr fd_io = algo::feature_descriptor_io::create("core");
  if (!fd_io)
  {
    TEST_ERROR("Unable to create core::feature_descriptor_io by name");
  }
}


// test writing Null data, which should throw an exception
IMPLEMENT_TEST(write_null_features_descriptors)
{
  using namespace kwiver::arrows;

  feature_set_sptr empty_features;
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  EXPECT_EXCEPTION(kwiver::vital::invalid_value,
                   fd_io.save("temp.kwfd", empty_features, empty_descriptors),
                   "attempting to save a file with Null features and descriptors");

  //TEST_EQUAL("All points are inliers", num_inliers, norm_pts1.size());
}


// test writing Null data, which should throw an exception
IMPLEMENT_TEST(load_bad_file_path)
{
  using namespace kwiver::arrows;

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;

  core::feature_descriptor_io fd_io;
  EXPECT_EXCEPTION(kwiver::vital::path_not_exists,
                   fd_io.load("invalid_file_path.kwfd", loaded_features, loaded_descriptors),
                   "attempting to load from an invalid file path");

  //TEST_EQUAL("All points are inliers", num_inliers, norm_pts1.size());
}


template <typename T>
feature_set_sptr
make_n_features(size_t num_feat)
{
  std::vector<feature_sptr> feat;
  for(unsigned i=0; i<num_feat; ++i)
  {
    T v = static_cast<T>(i) / num_feat;
    auto f = std::make_shared<feature_<T> >();
    T x = v * 10, y = v * 15 + 5;
    f->set_loc(vector_2f(x,y));
    f->set_scale(1.0 + v);
    f->set_magnitude(1 - v);
    f->set_angle(v * 3.14159f);
    f->set_color( rgb_color(static_cast<uint8_t>(i),
                            static_cast<uint8_t>(i+5),
                            static_cast<uint8_t>(i+10)) );
    f->set_covar( covariance_<2, T>(v) );
    feat.push_back(f);
  }

  return std::make_shared<simple_feature_set>(feat);
}


// test writing just features
IMPLEMENT_TEST(read_write_features)
{
  using namespace kwiver::arrows;

  feature_set_sptr features = make_n_features<float>(100);
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  fd_io.save("temp.kwfd", features, empty_descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load("temp.kwfd", loaded_features, loaded_descriptors);
}
