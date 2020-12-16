// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/ocv/estimate_fundamental_matrix.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;

using ocv::estimate_fundamental_matrix;

static constexpr double ideal_tolerance = 1e-6;
static constexpr double outlier_tolerance = 0.01;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(estimate_fundamental_matrix, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE( nullptr, algo::estimate_fundamental_matrix::create("ocv") );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_estimate_fundamental_matrix.h>
