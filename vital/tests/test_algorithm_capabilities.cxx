// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test capabilities class
 */

#include <vital/algorithm_capabilities.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class algorithm_capabilities : public ::testing::Test
{
public:
  void SetUp()
  {
    cap.set_capability( "cap1", true );
    cap.set_capability( "cap2", false );
  }

  kwiver::vital::algorithm_capabilities cap;
};

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, empty)
{
  kwiver::vital::algorithm_capabilities cap_empty;

  EXPECT_EQ( false, cap_empty.has_capability( "test" ) );

  auto cap_list = cap_empty.capability_list();
  EXPECT_TRUE( cap_list.empty() );
  EXPECT_EQ( 0, cap_list.size() );
}

// ----------------------------------------------------------------------------
static void test_capabilities(
  kwiver::vital::algorithm_capabilities const& cap )
{
  auto cap_list = cap.capability_list();
  EXPECT_EQ( 2, cap_list.size() );

  EXPECT_TRUE( cap.has_capability( "cap1" ) );
  EXPECT_TRUE( cap.has_capability( "cap2" ) );

  EXPECT_EQ( true, cap.capability( "cap1" ) );
  EXPECT_EQ( false, cap.capability( "cap2" ) );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, api)
{
  test_capabilities( cap );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, copy)
{
  kwiver::vital::algorithm_capabilities cap_copied( cap );
  test_capabilities( cap_copied );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, assign)
{
  kwiver::vital::algorithm_capabilities cap_assigned;
  cap_assigned = cap;
  test_capabilities( cap_assigned );
}
