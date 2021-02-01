// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_tmpfn.h>

#include <arrows/core/feature_descriptor_io.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

#include <cstdlib>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(feature_descriptor_io, create)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  EXPECT_NE(nullptr, algo::feature_descriptor_io::create("core"));
}

// ----------------------------------------------------------------------------
// Test writing Null data, which should throw an exception
TEST(feature_descriptor_io, write_null_features_descriptors)
{
  using namespace kwiver::arrows;

  feature_set_sptr empty_features;
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  EXPECT_THROW(
    fd_io.save("temp.kwfd", empty_features, empty_descriptors),
    kwiver::vital::invalid_value)
    << "Attempting to save a file with Null features and descriptors";
}

// ----------------------------------------------------------------------------
// Test writing Null data, which should throw an exception
TEST(feature_descriptor_io, load_bad_file_path)
{
  using namespace kwiver::arrows;

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;

  core::feature_descriptor_io fd_io;
  EXPECT_THROW(
    fd_io.load("invalid_file_path.kwfd", loaded_features, loaded_descriptors),
    kwiver::vital::path_not_exists)
    << "Attempting to load from an invalid file path";
}

namespace {

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
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

  for(unsigned i = 0; i < num_desc; ++i)
  {
    auto d = std::make_shared<descriptor_dynamic<T> >(dim);
    T* data = d->raw_data();
    for(unsigned j = 0; j < dim; ++j, ++data)
    {
      *data = static_cast<T>(rand() * scale + tmin );
    }
    desc.push_back(d);
  }

  return std::make_shared<simple_descriptor_set>(desc);
}

// ----------------------------------------------------------------------------
::testing::AssertionResult
equal_feature_set(feature_set_sptr fs1, feature_set_sptr fs2)
{
  if( !fs1 && !fs2 )
  {
    return ::testing::AssertionSuccess();
  }
  if( !fs1 || !fs2 )
  {
    return ::testing::AssertionFailure() << "empty / non-empty mismatch";
  }
  if( fs1->size() != fs2->size() )
  {
    return ::testing::AssertionFailure() << "size mismatch";
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
      return ::testing::AssertionFailure()
        << "feature existence mismatch at index " << i;
    }
    if( *f1 != *f2 )
    {
      return ::testing::AssertionFailure()
        << "features at index " << i << " are inequal";
    }
  }
  return ::testing::AssertionSuccess();
}

// ----------------------------------------------------------------------------
::testing::AssertionResult
equal_descriptor_set(descriptor_set_sptr ds1, descriptor_set_sptr ds2)
{
  if( !ds1 && !ds2 )
  {
    return ::testing::AssertionSuccess();
  }
  if( !ds1 || !ds2 )
  {
    return ::testing::AssertionFailure() << "empty / non-empty mismatch";
  }
  if( ds1->size() != ds2->size() )
  {
    return ::testing::AssertionFailure() << "size mismatch";
  }
  for( unsigned i=0; i< ds1->size(); ++i )
  {
    descriptor_sptr d1 = ds1->at(i);
    descriptor_sptr d2 = ds2->at(i);
    if( !d1 && !d2 )
    {
      continue;
    }
    if( !d1 || !d2 )
    {
      return ::testing::AssertionFailure()
        << "descriptor existence mismatch at index " << i;
    }
    if( *d1 != *d2 )
    {
      return ::testing::AssertionFailure()
        << "descriptors at index " << i << " are inequal";
    }
  }
  return ::testing::AssertionSuccess();
}

}

// ----------------------------------------------------------------------------
// Test writing just features
TEST(feature_descriptor_io, read_write_features)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  feature_set_sptr features = make_n_features<float>(100);
  descriptor_set_sptr empty_descriptors;

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, descriptor_set_sptr{});

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  EXPECT_EQ(nullptr, loaded_descriptors);
  EXPECT_TRUE(equal_feature_set(features, loaded_features));
  EXPECT_EQ(0, std::remove(filename.c_str()));
}

// ----------------------------------------------------------------------------
// Test writing just descriptors
TEST(feature_descriptor_io, read_write_descriptors)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  descriptor_set_sptr descriptors = make_n_descriptors<float>(100, 128);

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, feature_set_sptr{}, descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  EXPECT_EQ(nullptr, loaded_features);
  EXPECT_TRUE(equal_descriptor_set(descriptors, loaded_descriptors));
  EXPECT_EQ(0, std::remove(filename.c_str()));
}

// ----------------------------------------------------------------------------
// Test writing both features and descriptors
TEST(feature_descriptor_io, read_write_features_descriptors)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  feature_set_sptr features = make_n_features<double>(50);
  descriptor_set_sptr descriptors = make_n_descriptors<uint8_t>(100, 96);

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, descriptors);

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  EXPECT_TRUE(equal_feature_set(features, loaded_features));
  EXPECT_TRUE(equal_descriptor_set(descriptors, loaded_descriptors));
  EXPECT_EQ(0, std::remove(filename.c_str()));
}

// ----------------------------------------------------------------------------
// Test writing a mix of features types
TEST(feature_descriptor_io, read_write_mixed_features)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  feature_set_sptr features1 = make_n_features<double>(50);
  feature_set_sptr features2 = make_n_features<float>(50);
  std::vector<feature_sptr> feat1 = features1->features();
  std::vector<feature_sptr> feat2 = features2->features();
  feat1.insert(feat1.end(), feat2.begin(), feat2.end());
  feature_set_sptr features = std::make_shared<simple_feature_set>(feat1);

  core::feature_descriptor_io fd_io;
  fd_io.save(filename, features, descriptor_set_sptr{});

  feature_set_sptr loaded_features;
  descriptor_set_sptr loaded_descriptors;
  fd_io.load(filename, loaded_features, loaded_descriptors);

  EXPECT_EQ(nullptr, loaded_descriptors);
  // This is false because the writer will convert all the features to a common
  // type, which ever comes first in the vector.
  EXPECT_FALSE(equal_feature_set(features, loaded_features));
  EXPECT_EQ(0, std::remove(filename.c_str()));
}

// ----------------------------------------------------------------------------
// Test writing a mix of descriptor dimensions
TEST(feature_descriptor_io, write_mixed_descriptor_dim)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  feature_set_sptr empty_features;
  descriptor_set_sptr descriptors1 = make_n_descriptors<int16_t>(50, 128);
  descriptor_set_sptr descriptors2 = make_n_descriptors<int16_t>(50, 64);
  std::vector<descriptor_sptr> desc1 = descriptors1->descriptors();
  desc1.insert(desc1.end(), descriptors2->cbegin(), descriptors2->cend());
  descriptor_set_sptr descriptors = std::make_shared<simple_descriptor_set>(desc1);
  core::feature_descriptor_io fd_io;
  EXPECT_THROW(fd_io.save(filename, empty_features, descriptors),
               kwiver::vital::invalid_data)
    << "Expected exception saving a mixture of descriptor dimensions";
  EXPECT_EQ(0, std::remove(filename.c_str()));
}

// ----------------------------------------------------------------------------
// Test writing a mix of descriptor types
TEST(feature_descriptor_io, write_mixed_descriptor_type)
{
  using namespace kwiver::arrows;
  auto const filename = kwiver::testing::temp_file_name( "test-", ".kwfd" );

  feature_set_sptr empty_features;
  descriptor_set_sptr descriptors1 = make_n_descriptors<uint16_t>(50, 96);
  descriptor_set_sptr descriptors2 = make_n_descriptors<uint32_t>(50, 96);
  std::vector<descriptor_sptr> desc1 = descriptors1->descriptors();
  desc1.insert(desc1.end(), descriptors2->cbegin(), descriptors2->cend());
  descriptor_set_sptr descriptors = std::make_shared<simple_descriptor_set>(desc1);
  core::feature_descriptor_io fd_io;
  EXPECT_THROW(fd_io.save(filename, empty_features, descriptors),
               kwiver::vital::invalid_data)
    << "Expected exception saving a mixture of descriptor types";
  EXPECT_EQ(0, std::remove(filename.c_str()));
}
