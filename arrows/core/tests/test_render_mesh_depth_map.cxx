/*ckwg +29
 * Copyright 2018 by Kitware, SAS.
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

#include <gtest/gtest.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/core/render_mesh_depth_map.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/image_container.h>
#include <vital/types/mesh.h>
#include <vital/types/vector.h>
#include <memory>

using namespace kwiver::vital;

namespace
{
  const vector_3d A(-5.0, -5.0, 0.0);
  const vector_3d B( 5.0, -5.0, 0.0);
  const vector_3d C( 0.0,  5.0, 1.0);

  const std::vector<vector_3d> vertices = {A, B, C};
  const mesh_regular_face<3> face = std::vector<unsigned int>({0, 1, 2});
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

kwiver::vital::mesh_sptr generate_mesh()
{
  std::unique_ptr<mesh_vertex_array_base> verts(new mesh_vertex_array<3>(vertices));
  std::unique_ptr<mesh_regular_face_array<3>> faces(new mesh_regular_face_array<3>());
  faces->push_back(face);
  mesh_sptr mesh(new kwiver::vital::mesh(std::move(verts), std::move(faces)));
  return mesh;
}


// ----------------------------------------------------------------------------
TEST(render_mesh_depth_map, perspective_camera)
{
  // Mesh
  mesh_sptr mesh = generate_mesh();

  // Perspective camera
  camera_intrinsics_sptr camera_intrinsic(new simple_camera_intrinsics(780, {500, 500},
                                                                       1.0, 0.0, {},
                                                                       1000, 1000));
  matrix_3x3d rot = matrix_3x3d::Zero();
  rot(0, 0) = 1.0;
  rot(1, 1) = -1.0;
  rot(2, 2) = -1.0;
  rotation_d cam_orientation(rot);
  vector_3d cam_position(0.0, 0.0, 10.0);
  camera_perspective_sptr camera(new simple_camera_perspective(cam_position,
                                                               cam_orientation.inverse(),
                                                               camera_intrinsic));

  image_container_sptr depth_map = kwiver::arrows::render_mesh_depth_map(mesh, camera);

  // Check barycenter;
  vector_3d barycenter = (A + B + C) / 3;

  double real_depth_A = camera->depth(A);
  double real_depth_B = camera->depth(B);
  double real_depth_C = camera->depth(C);
  double real_depth_bary = camera->depth(barycenter);

  vector_2d a = camera->project(A);
  vector_2d b = camera->project(B);
  vector_2d c = camera->project(C);
  vector_2d bary = camera->project(barycenter);

  double measured_depth_A = depth_map->get_image().at<double>(static_cast<int>(std::round(a(0))),
                                                              static_cast<int>(std::round(a(1))));

  double measured_depth_B = depth_map->get_image().at<double>(static_cast<int>(std::round(b(0))),
                                                              static_cast<int>(std::round(b(1))));

  double measured_depth_C = depth_map->get_image().at<double>(static_cast<int>(std::round(c(0))),
                                                              static_cast<int>(std::round(c(1))));

  double measured_depth_bary = depth_map->get_image().at<double>(static_cast<int>(std::round(bary(0))),
                                                                 static_cast<int>(std::round(bary(1))));

  EXPECT_NEAR( real_depth_A, measured_depth_A, 1e-2 );

  EXPECT_NEAR( real_depth_B, measured_depth_B, 1e-2 ) ;

  EXPECT_NEAR( real_depth_C, measured_depth_C, 1e-2 ) ;

  EXPECT_NEAR( real_depth_bary, measured_depth_bary, 1e-2 ) ;
}
