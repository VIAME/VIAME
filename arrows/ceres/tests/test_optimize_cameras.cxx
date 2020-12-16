// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/ceres/optimize_cameras.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

using kwiver::arrows::ceres::optimize_cameras;

#if defined(_MSC_VER)
static constexpr double noisy_center_tolerance = 1e-8;
static constexpr double noisy_rotation_tolerance = 2e-9;
static constexpr double noisy_intrinsics_tolerance = 2e-6;
#elif defined(__clang__)
static constexpr double noisy_center_tolerance = 1e-8;
static constexpr double noisy_rotation_tolerance = 1e-9;
static constexpr double noisy_intrinsics_tolerance = 5e-6;
#else
static constexpr double noisy_center_tolerance = 1e-9;
static constexpr double noisy_rotation_tolerance = 1e-11;
static constexpr double noisy_intrinsics_tolerance = 1e-7;
#endif

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(optimize_cameras, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE( nullptr, algo::optimize_cameras::create("ceres") );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_optimize_cameras.h>
