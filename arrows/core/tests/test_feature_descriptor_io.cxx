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

#include <cstdlib>

#include <arrows/core/feature_descriptor_io.h>
#include <vital/plugin_loader/plugin_manager.h>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  kwiver::vital::plugin_manager::instance().load_all_plugins();

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

namespace {

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
    f->set_loc(Eigen::Matrix<T, 2, 1>(x,y));
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


template <typename T>
descriptor_set_sptr
make_n_descriptors(size_t num_desc, size_t dim)
{
  std::vector<descriptor_sptr> desc;
  const double rmax = static_cast<double>(RAND_MAX);
  const double tmin = std::numeric_limits<T>::is_integer
                    ? static_cast<double>(std::numeric_limits<T>::min()) : 0.0;
  const double tmax = std::numeric_limits<T>::is_integer
                    ? static_cast<double>(std::numeric_limits<T>::max()) : 1.0;
  const double scale = (tmax - tmin) / rmax;

  for(unsigned i=0; i<num_desc; ++i)
  {
    auto d = std::make_shared<descriptor_dynamic<T> >(dim);
    T* data = d->raw_data();
    for(unsigned i=0; i<dim; ++i, ++data)
    {
      *data = static_cast<T>(rand() * scale + tmin );
    }
    desc.push_back(d);
  }

  return std::make_shared<simple_descriptor_set>(desc);
}


bool equal_feature_set(feature_set_sptr fs1, feature_set_sptr fs2)
{
  if( !fs1 && !fs2 )
  {
    return true;
  }
  if( !fs1 || !fs2 )
  {
    return false;
  }
  if( fs1->size() != fs2->size() )
  {
    return false;
  }
  std::vector<feature_sptr> feat1 = fs1->features();
  std::vector<feature_sptr> feat2 = fs2->features();
  for( unsigned i=0; i< feat1.size(); ++i )
  {
    feature_sptr f1 = feat1[i];
    feature_sptr f2 = feat2[i];
    if( !f1 && !f2 )
    {
      continue;
    }
    if( !f1 || !f2 )
    {
      return false;
    }
    if( *f1 != *f2 )
    {
      return false;
    }
  }
  return true;
}


bool equal_descriptor_set(descriptor_set_sptr ds1, descriptor_set_sptr ds2)
{
  if( !ds1 && !ds2 )
  {
    return true;
  }
  if( !ds1 || !ds2 )
  {
    return false;
  }
  if( ds1->size() != ds2->size() )
  {
    return false;
  }
  std::vector<descriptor_sptr> desc1 = ds1->descriptors();
  std::vector<descriptor_sptr> desc2 = ds2->descriptors();
  for( unsigned i=0; i< desc1.size(); ++i )
  {
    descriptor_sptr d1 = desc1[i];
    descriptor_sptr d2 = desc2[i];
    if( !d1 && !d2 )
    {
      continue;
    }
    if( !d1 || !d2 )
    {
      return false;
    }
    if( *d1 != *d2 )
    {
      return false;
    }
  }
  return true;
}


}


// test writing just features
IMPLEMENT_TEST(read_write_features)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr features = make_n_features<float>(100);
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, empty_descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  TEST_EQUAL("empty descriptors", empty_descriptors, loaded_descriptors);
  TEST_EQUAL("compare features", equal_feature_set(features, loaded_features), true);
  std::remove(filename);
}


// test writing just descriptors
IMPLEMENT_TEST(read_write_descriptors)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr empty_features;
  descriptor_set_sptr descriptors = make_n_descriptors<float>(100, 128);

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, empty_features, descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  TEST_EQUAL("empty features", empty_features, loaded_features);
  TEST_EQUAL("compare descriptors", equal_descriptor_set(descriptors, loaded_descriptors), true);
  std::remove(filename);
}


// test writing both features and descriptors
IMPLEMENT_TEST(read_write_features_descriptors)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr features = make_n_features<double>(50);
  descriptor_set_sptr descriptors = make_n_descriptors<uint8_t>(100, 96);

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  TEST_EQUAL("compare features", equal_feature_set(features, loaded_features), true);
  TEST_EQUAL("compare descriptors", equal_descriptor_set(descriptors, loaded_descriptors), true);
  std::remove(filename);
}


// test writing a mix of features types
IMPLEMENT_TEST(read_write_mixed_features)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr features1 = make_n_features<double>(50);
  feature_set_sptr features2 = make_n_features<float>(50);
  std::vector<feature_sptr> feat1 = features1->features();
  std::vector<feature_sptr> feat2 = features2->features();
  feat1.insert(feat1.end(), feat2.begin(), feat2.end());
  feature_set_sptr features = std::make_shared<simple_feature_set>(feat1);
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, empty_descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  TEST_EQUAL("empty descriptors", empty_descriptors, loaded_descriptors);
  // this is false because the writer will convert all the features to a common type,
  // which ever comes first in the vector.
  TEST_EQUAL("compare features", equal_feature_set(features, loaded_features), false);
  std::remove(filename);
}


// test writing a mix of descriptor dimensions
IMPLEMENT_TEST(write_mixed_descriptor_dim)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr empty_features;
  descriptor_set_sptr descriptors1 = make_n_descriptors<int16_t>(50, 128);
  descriptor_set_sptr descriptors2 = make_n_descriptors<int16_t>(50, 64);
  std::vector<descriptor_sptr> desc1 = descriptors1->descriptors();
  std::vector<descriptor_sptr> desc2 = descriptors2->descriptors();
  desc1.insert(desc1.end(), desc2.begin(), desc2.end());
  descriptor_set_sptr descriptors = std::make_shared<simple_descriptor_set>(desc1);
  core::feature_descriptor_io fd_io;
  EXPECT_EXCEPTION(kwiver::vital::invalid_data,
                   fd_io.save(filename, empty_features, descriptors),
                   "cannot save a mixture of descriptor dimensions");
  std::remove(filename);
}


// test writing a mix of descriptor types
IMPLEMENT_TEST(write_mixed_descriptor_type)
{
  using namespace kwiver::arrows;
  const char filename[] = "temp.kwfd";

  feature_set_sptr empty_features;
  descriptor_set_sptr descriptors1 = make_n_descriptors<uint16_t>(50, 96);
  descriptor_set_sptr descriptors2 = make_n_descriptors<uint32_t>(50, 96);
  std::vector<descriptor_sptr> desc1 = descriptors1->descriptors();
  std::vector<descriptor_sptr> desc2 = descriptors2->descriptors();
  desc1.insert(desc1.end(), desc2.begin(), desc2.end());
  descriptor_set_sptr descriptors = std::make_shared<simple_descriptor_set>(desc1);
  core::feature_descriptor_io fd_io;
  EXPECT_EXCEPTION(kwiver::vital::invalid_data,
                   fd_io.save(filename, empty_features, descriptors),
                   "cannot save a mixture of descriptor types");
  std::remove(filename);
}
