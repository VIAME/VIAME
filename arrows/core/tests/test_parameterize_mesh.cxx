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


#include <arrows/core/parameterize_mesh.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/mesh.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace kwiver::vital;
using namespace kwiver::arrows;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  plugin_manager::instance().load_all_plugins();
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

class compute_mesh_uv_parameterization : public ::testing::Test
{
public:
  void SetUp()
  {
    // cube mesh of size 1.0
    std::vector<vector_3d> verts = {
          {-0.500000, -0.500000, -0.500000},
          {-0.500000, -0.500000, 0.500000},
          {-0.500000, 0.500000, -0.500000},
          {-0.500000, 0.500000, 0.500000},
          {0.500000 ,-0.500000, -0.500000},
          {0.500000 ,-0.500000, 0.500000},
          {0.500000 ,0.500000 ,-0.500000},
          {0.500000 ,0.500000 ,0.500000}
    };
    std::vector< mesh_regular_face<3> > faces;
    faces.push_back(mesh_regular_face<3>({0, 1, 2}));
    faces.push_back(mesh_regular_face<3>({3, 2, 1}));
    faces.push_back(mesh_regular_face<3>({4, 6, 5}));
    faces.push_back(mesh_regular_face<3>({7, 5, 6}));
    faces.push_back(mesh_regular_face<3>({0, 4, 1}));
    faces.push_back(mesh_regular_face<3>({5, 1, 4}));
    faces.push_back(mesh_regular_face<3>({2, 3, 6}));
    faces.push_back(mesh_regular_face<3>({7, 6, 3}));
    faces.push_back(mesh_regular_face<3>({0, 2, 4}));
    faces.push_back(mesh_regular_face<3>({6, 4, 2}));
    faces.push_back(mesh_regular_face<3>({1, 5, 3}));
    faces.push_back(mesh_regular_face<3>({7, 3, 5}));

    std::unique_ptr<mesh_vertex_array_base> vertices_array_ptr(new mesh_vertex_array<3>(verts));
    std::unique_ptr<mesh_face_array_base> faces_array_ptr(new mesh_regular_face_array<3>(faces));
    mesh = std::make_shared<kwiver::vital::mesh>(std::move(vertices_array_ptr), std::move(faces_array_ptr));
  }

  mesh_sptr mesh;
};


// ----------------------------------------------------------------------------
TEST_F(compute_mesh_uv_parameterization, check_texture_coordinates)
{
  double resolution = 0.03;   // mesh unit/pixel
  unsigned int interior_margin = 2;
  unsigned int exterior_margin = 2;

  kwiver::arrows::core::parameterize_mesh uv_param;
  kwiver::vital::config_block_sptr algo_config = uv_param.get_configuration();
  algo_config->set_value<double>("resolution", resolution);
  algo_config->set_value<unsigned int>("interior_margin", interior_margin);
  algo_config->set_value<unsigned int>("exterior_margin", exterior_margin);
  uv_param.set_configuration(algo_config);

  uv_param.parameterize(mesh);

  // check that texture coordinates are between 0 and 1
  const std::vector<vector_2d>& tcoords = mesh->tex_coords();
  for (auto tc: tcoords)
  {
    EXPECT_GE(tc[0], 0.0);
    EXPECT_GE(tc[1], 0.0);
    EXPECT_LE(tc[0], 1.0);
    EXPECT_LE(tc[1], 1.0);
  }
}

// ----------------------------------------------------------------------------
TEST_F(compute_mesh_uv_parameterization, check_faces_area)
{
  double resolution = 0.03;   // mesh unit/pixel
  unsigned int interior_margin = 2;
  unsigned int exterior_margin = 2;

  kwiver::arrows::core::parameterize_mesh uv_param;

  kwiver::vital::config_block_sptr algo_config = uv_param.get_configuration();
  algo_config->set_value<double>("resolution", resolution);
  algo_config->set_value<unsigned int>("interior_margin", interior_margin);
  algo_config->set_value<unsigned int>("exterior_margin", exterior_margin);
  uv_param.set_configuration(algo_config);

  std::pair<unsigned int, unsigned int> atlas_dim = uv_param.parameterize(mesh);

  const std::vector<vector_2d>& tcoords = mesh->tex_coords();
  // check that the faces area are preserved (with a resolution factor)
  for (unsigned int i = 0; i < tcoords.size(); i += 3)
  {
      vector_2d v1 = tcoords[i + 1] - tcoords[i + 0];
      vector_2d v2 = tcoords[i + 2] - tcoords[i + 0];
      v1 = v1.cwiseProduct(vector_2d(atlas_dim.first, atlas_dim.second));
      v2 = v2.cwiseProduct(vector_2d(atlas_dim.first, atlas_dim.second));
      double triangle_area = (std::abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2) * (resolution * resolution);
      EXPECT_NEAR(triangle_area, 0.5, 1e-5);
  }
}
