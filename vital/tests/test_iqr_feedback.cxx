// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test iqr_feedback class
 */

#include <vital/types/iqr_feedback.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

namespace {

std::vector<unsigned> const positive_samples  = { 2, 5, 6, 7, 8 };
std::vector<unsigned> const negative_samples  = { 1, 3, 4 };

}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(iqr_feedback, ensure_values)
{
  iqr_feedback feedback;

  feedback.set_positive_ids( positive_samples );
  feedback.set_negative_ids( negative_samples );

  EXPECT_EQ( positive_samples, feedback.positive_ids() );
  EXPECT_EQ( negative_samples, feedback.negative_ids() );
}
