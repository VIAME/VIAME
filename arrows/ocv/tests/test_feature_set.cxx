// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test OCV feature_set class
 */

#include <test_common.h>

#include <arrows/ocv/feature_set.h>

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
TEST(feature_set, default_set)
{
  ocv::feature_set fs;
  if (fs.size() != 0)
  {
    TEST_ERROR("Default features_set is not empty");
  }
  if (!fs.features().empty())
  {
    TEST_ERROR("Default features_set produces non-empty features");
  }
}

// ----------------------------------------------------------------------------
// It seems operator== on cv::Keypoint is not defined in OpenCV
static bool keypoints_equal(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2)
{
  return kp1.angle == kp2.angle &&
         kp1.class_id == kp2.class_id &&
         kp1.octave == kp2.octave &&
         kp1.pt == kp2.pt &&
         kp1.response == kp2.response &&
         kp1.size == kp2.size;
}

// ----------------------------------------------------------------------------
TEST(feature_set, populated_set)
{
  static constexpr unsigned num_feat = 100;

  std::vector<cv::KeyPoint> kpts;
  for ( unsigned i = 0; i < num_feat; ++i )
  {
    cv::KeyPoint kp(i/2.0f, i/3.0f, i/10.0f, (i*3.14159f)/num_feat, 100.0f/i);
    kpts.push_back(kp);
  }

  ocv::feature_set fs(kpts);
  EXPECT_EQ( num_feat, fs.size() );

  std::vector<cv::KeyPoint> kpts2 = fs.ocv_keypoints();
  EXPECT_TRUE( std::equal( kpts.begin(), kpts.end(),
                           kpts2.begin(), keypoints_equal ) );

  std::vector<feature_sptr> feats = fs.features();
  EXPECT_EQ( fs.size(), feats.size() );

  [&]{
    for ( unsigned i = 0; i < num_feat; ++i )
    {
      SCOPED_TRACE( "At feature " + std::to_string(i) );
      ASSERT_EQ( typeid(float), feats[i]->data_type() );
    }
  }();

  simple_feature_set simp_fs(feats);
  kpts2 = ocv::features_to_ocv_keypoints(simp_fs);
  EXPECT_TRUE( std::equal( kpts.begin(), kpts.end(),
                           kpts2.begin(), keypoints_equal ) )
    << "Conversion to and from ARROWS features should preserve cv::KeyPoints";
}
