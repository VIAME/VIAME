/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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

#include <test_scene.h>

#include <arrows/tests/test_triangulate_landmarks.h>
#include <arrows/vxl/triangulate_landmarks.h>

#include <vital/plugin_loader/plugin_manager.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(triangulate_landmarks, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE(nullptr, algo::triangulate_landmarks::create("vxl"));
}

// ----------------------------------------------------------------------------
// Input to triangulation is the ideal solution, make sure it doesn't diverge
TEST(triangulate_landmarks, from_solution)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_from_solution(tri_lm);
}

// ----------------------------------------------------------------------------
// Add noise to landmarks before input to triangulation
TEST(triangulate_landmarks, noisy_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_noisy_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Initialize all landmarks to the origin as input to triangulation
TEST(triangulate_landmarks, zero_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_zero_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of cameras to triangulation from
TEST(triangulate_landmarks, subset_cameras)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_cameras(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of landmarks to triangulate
TEST(triangulate_landmarks, subset_landmarks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_landmarks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states to constrain the problem
TEST(triangulate_landmarks, subset_tracks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_subset_tracks(tri_lm);
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states and add noise
TEST(triangulate_landmarks, noisy_tracks)
{
  kwiver::arrows::vxl::triangulate_landmarks tri_lm;
  kwiver::testing::test_noisy_tracks(tri_lm);
}
