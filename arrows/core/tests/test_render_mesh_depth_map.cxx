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
#include <vital/types/camera_rpc.h>
#include <vital/types/image_container.h>
#include <vital/types/mesh.h>
#include <vital/types/vector.h>
#include <memory>

using namespace kwiver::vital;

namespace
{
  const vector_3d A(-5.0, -5.0, -2);
  const vector_3d B( 5.0, -5.0, -2);
  const vector_3d C( 0.0,  5.0, 1.0);

  const vector_3d D(-5.5, -5.5, -1.0);
  const vector_3d E(-3.0, -5.5, -1.0);
  const vector_3d F(-5.0, -3.0, -1.0);

  const std::vector<vector_3d> vertices = {A, B, C, D, E, F};
  const mesh_regular_face<3> face1 = std::vector<unsigned int>({0, 1, 2});
  const mesh_regular_face<3> face2 = std::vector<unsigned int>({3, 4, 5});

  double check_neighbouring_pixels(const kwiver::vital::image& image, double x, double y)
  {
    /// check the four closest pixels and return the average of the finite values
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);

    double v1;
    double sum = 0.0;
    int nb = 0;

    if (x1 >= 0 && x1 < static_cast<int>(image.width()) && y1 >= 0 && y1 < static_cast<int>(image.height()))
    {
      v1 = image.at<double>(x1, y1);
      if (!std::isinf(v1))
      {
        sum += v1;
        nb++;
      }
    }
    ++x1;
    if (x1 >= 0 && x1 < static_cast<int>(image.width()) && y1 >= 0 && y1 < static_cast<int>(image.height()))
    {
      v1 = image.at<double>(x1, y1);
      if (!std::isinf(v1))
      {
        sum += v1;
        nb++;
      }
    }
    ++y1;
    if (x1 >= 0 && x1 < static_cast<int>(image.width()) && y1 >= 0 && y1 < static_cast<int>(image.height()))
    {
      v1 = image.at<double>(x1, y1);
      if (!std::isinf(v1))
      {
        sum += v1;
        nb++;
      }
    }
    --x1;
    if (x1 >= 0 && x1 < static_cast<int>(image.width()) && y1 >= 0 && y1 < static_cast<int>(image.height()))
    {
      v1 = image.at<double>(x1, y1);
      if (!std::isinf(v1))
      {
        sum += v1;
        nb++;
      }
    }
    if (nb > 0)
      return sum / nb;
    else
      return std::numeric_limits<double>::infinity();
  }
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
  faces->push_back(face1);
  faces->push_back(face2);
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

  image_container_sptr depth_map = kwiver::arrows::core::render_mesh_depth_map(mesh, camera);

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

  double measured_depth_A = check_neighbouring_pixels(depth_map->get_image(), a(0), a(1));
  double measured_depth_B = check_neighbouring_pixels(depth_map->get_image(), b(0), b(1));
  double measured_depth_C = check_neighbouring_pixels(depth_map->get_image(), c(0), c(1));
  double measured_depth_bary = check_neighbouring_pixels(depth_map->get_image(), bary(0), bary(1));

  EXPECT_LT( measured_depth_A, real_depth_A );    // A is hidden by face DEF

  EXPECT_NEAR( real_depth_B, measured_depth_B, 1e-2 );

  EXPECT_NEAR( real_depth_C, measured_depth_C, 1e-2 );

  EXPECT_NEAR( real_depth_bary, measured_depth_bary, 1e-2 );
}


// ----------------------------------------------------------------------------
TEST(render_mesh_height_map, perspective_camera)
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

  image_container_sptr height_map = kwiver::arrows::core::render_mesh_height_map(mesh, camera);

  // Check barycenter;
  vector_3d barycenter = (A + B + C) / 3;

  double real_height_A = A.z();
  double real_height_B = B.z();
  double real_height_C = C.z();
  double real_height_bary = barycenter.z();

  vector_2d a = camera->project(A);
  vector_2d b = camera->project(B);
  vector_2d c = camera->project(C);
  vector_2d bary = camera->project(barycenter);

  double measured_height_A = check_neighbouring_pixels(height_map->get_image(), a(0), a(1));
  double measured_height_B = check_neighbouring_pixels(height_map->get_image(), b(0), b(1));
  double measured_height_C = check_neighbouring_pixels(height_map->get_image(), c(0), c(1));
  double measured_height_bary = check_neighbouring_pixels(height_map->get_image(), bary(0), bary(1));

  EXPECT_GT( measured_height_A, real_height_A );    // A is hidden by face DEF

  EXPECT_NEAR( real_height_B, measured_height_B, 1e-2 );

  EXPECT_NEAR( real_height_C, measured_height_C, 1e-2 );

  EXPECT_NEAR( real_height_bary, measured_height_bary, 1e-2 );
}


TEST(render_mesh_height_map, camera_rpc)
{
  // Mesh
  mesh_sptr mesh = generate_mesh();

  // Camera RPC
  std::shared_ptr<simple_camera_rpc> camera(new simple_camera_rpc);
  camera->set_image_scale(kwiver::vital::vector_2d(100, 100));
  camera->set_image_offset(kwiver::vital::vector_2d(550, 550));
  camera->set_image_width(1200);
  camera->set_image_height(1200);

  image_container_sptr height_map = kwiver::arrows::core::render_mesh_height_map(mesh, camera);

  // Check barycenter;
  vector_3d barycenter = (A + B + C) / 3;

  double real_height_A = A.z();
  double real_height_B = B.z();
  double real_height_C = C.z();
  double real_height_bary = barycenter.z();

  vector_2d a = camera->project(A);
  vector_2d b = camera->project(B);
  vector_2d c = camera->project(C);
  vector_2d bary = camera->project(barycenter);

  double measured_height_A = check_neighbouring_pixels(height_map->get_image(), a(0), a(1));
  double measured_height_B = check_neighbouring_pixels(height_map->get_image(), b(0), b(1));
  double measured_height_C = check_neighbouring_pixels(height_map->get_image(), c(0), c(1));
  double measured_height_bary = check_neighbouring_pixels(height_map->get_image(), bary(0), bary(1));

  EXPECT_GT(measured_height_A, real_height_A);    // A is hidden by face DEF

  EXPECT_NEAR( real_height_B, measured_height_B, 1e-2 );

  EXPECT_NEAR( real_height_C, measured_height_C, 1e-2 );

  EXPECT_NEAR( real_height_bary, measured_height_bary, 1e-2 );
}
