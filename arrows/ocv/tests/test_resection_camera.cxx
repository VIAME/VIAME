// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/ocv/resection_camera.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;
using namespace std;

using ocv::resection_camera;

static constexpr double
  ideal_rotation_tolerance = 1e-6,
  ideal_center_tolerance = 1e-6,
  noisy_rotation_tolerance = 0.008,
  noisy_center_tolerance = 0.05,
  outlier_rotation_tolerance = 0.008,
  outlier_center_tolerance = 0.05;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( resection_camera, create )
{
  plugin_manager::instance().load_all_plugins();
  EXPECT_NE( nullptr, algo::resection_camera::create( "ocv" ) );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_resection_camera.h>
