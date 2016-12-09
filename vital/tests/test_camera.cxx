/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * \brief test core camera class
 */

#include <test_common.h>

#include <iostream>
#include <vital/types/camera.h>
#include <vital/io/camera_io.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(clone)
{
  using namespace kwiver::vital;
  vector_2d pp(300,400);
  simple_camera_intrinsics K(1000, pp);
  kwiver::vital::simple_camera cam(vector_3d(3, -4, 7), rotation_d(), K);

  auto cam2 = cam.clone();
  TEST_EQUAL("Center should be the same object after clone",
             cam.center(), cam2->center());
  TEST_EQUAL("Rotation should be the same object after clone",
             cam.rotation(), cam2->rotation());
  TEST_EQUAL("Instrinics should be the same object after clone",
             cam.intrinsics(), cam2->intrinsics());
}


IMPLEMENT_TEST(clone_look_at)
{
  using namespace kwiver::vital;
  vector_2d pp(300,400);
  simple_camera_intrinsics K(1000, pp);
  vector_3d focus(0, 1, -2);

  auto cam =
    std::shared_ptr<camera>(
      new simple_camera( vector_3d( 3, -4, 7 ), rotation_d(), K ) )
      ->clone_look_at( focus );

  vector_2d ifocus = cam->project(focus);
  TEST_NEAR("clone_look_at focus projects to origin",
            (ifocus-pp).norm(), 0.0, 1e-12);

  vector_2d ifocus_up = cam->project(focus + vector_3d(0,0,2));
  vector_2d tmp = ifocus_up - pp;
  TEST_NEAR("clone_look_at vertical projects vertical",
            tmp.x(), 0.0, 1e-12);
  // "up" in image space is actually negative Y because the
  // Y axis is inverted
  TEST_EQUAL("clone_look_at up projects up", tmp.y() < 0.0, true);
}


IMPLEMENT_TEST(look_at)
{
  using namespace kwiver::vital;
  vector_2d pp(300,400);
  simple_camera_intrinsics K(1000, pp);
  vector_3d focus(0, 1, -2);
  kwiver::vital::simple_camera cam(vector_3d(3, -4, 7), rotation_d(), K);
  cam.look_at( focus );

  vector_2d ifocus = cam.project(focus);
  TEST_NEAR("look_at focus projects to origin",
            (ifocus-pp).norm(), 0.0, 1e-12);

  vector_2d ifocus_up = cam.project(focus + vector_3d(0,0,2));
  vector_2d tmp = ifocus_up - pp;
  TEST_NEAR("look_at vertical projects vertical",
            tmp.x(), 0.0, 1e-12);
  // "up" in image space is actually negative Y because the
  // Y axis is inverted
  TEST_EQUAL("look_at up projects up", tmp.y() < 0.0, true);
}


IMPLEMENT_TEST(projection)
{
  using namespace kwiver::vital;
  vector_2d pp(300,400);
  simple_camera_intrinsics K(1000, pp);
  vector_3d focus(0, 1, -2);
  kwiver::vital::simple_camera cam(vector_3d(3, -4, 7), rotation_d(), K);
  cam.look_at( focus );

  matrix_3x4d P(cam.as_matrix());
  vector_3d test_pt(1,2,3);
  vector_4d test_hpt(test_pt.x(), test_pt.y(), test_pt.z(), 1.0);

  vector_3d proj_hpt = P * test_hpt;
  vector_2d proj_pt(proj_hpt.x()/proj_hpt.z(), proj_hpt.y()/proj_hpt.z());

  TEST_NEAR("camera projection = matrix multiplication",
             (cam.project(test_pt) - proj_pt).norm(), 0.0, 1e-12);
}
