// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test OCV match set class
 */

#include <arrows/ocv/match_set.h>

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
TEST(match_set, default_set)
{
  ocv::match_set ms;
  EXPECT_EQ( 0, ms.size() );
  EXPECT_TRUE( ms.matches().empty() );
}

// ----------------------------------------------------------------------------
// It seems operator== on cv::DMatch is not defined in OpenCV
static bool dmatch_equal(const cv::DMatch& dm1, const cv::DMatch& dm2)
{
  return dm1.queryIdx == dm2.queryIdx &&
         dm1.trainIdx == dm2.trainIdx &&
         dm1.imgIdx == dm2.imgIdx &&
         dm1.distance == dm2.distance;
}

// ----------------------------------------------------------------------------
TEST(match_set, populated_set)
{
  static constexpr unsigned num_matches = 100;

  std::vector<cv::DMatch> dms;
  for (unsigned i=0; i<num_matches; ++i)
  {
    cv::DMatch dm(i, num_matches-i-1, FLT_MAX);
    dms.push_back(dm);
  }

  ocv::match_set ms(dms);
  EXPECT_EQ( num_matches, ms.size() );

  std::vector<cv::DMatch> dms2 = ms.ocv_matches();
  EXPECT_TRUE( std::equal( dms.begin(), dms.end(),
                           dms2.begin(), dmatch_equal ) );

  std::vector<match> mats = ms.matches();
  EXPECT_EQ( ms.size(), mats.size() );

  simple_match_set simp_ms(mats);
  dms2 = ocv::matches_to_ocv_dmatch(simp_ms);
  EXPECT_TRUE( std::equal( dms.begin(), dms.end(),
                           dms2.begin(), dmatch_equal ) )
    << "Conversion to and from ARROWS features should preserve cv::DMatch";
}
