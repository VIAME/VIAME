// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL homography estimation algorithm
 */

#include <test_eigen.h>
#include <test_random_point.h>

#include <arrows/vxl/estimate_homography.h>

#include <vital/plugin_loader/plugin_manager.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;
using namespace kwiver::testing;

using vxl::estimate_homography;

static constexpr double ideal_matrix_tolerance = 1e-9;
static constexpr double ideal_norm_tolerance = 1e-8;

static constexpr double noisy_matrix_tolerance = 0.05;
static constexpr double noisy_norm_tolerance = 0.2;

static constexpr double outlier_matrix_tolerance = 1e-11;
static constexpr double outlier_norm_tolerance = 1e-8;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(estimate_homography, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE( nullptr, algo::estimate_homography::create("vxl") );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_estimate_homography.h>
