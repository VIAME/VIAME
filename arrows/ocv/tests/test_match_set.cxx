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
