// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/core/uv_unwrap_mesh.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/mesh.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace kwiver::vital;
using namespace kwiver::arrows::core;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

class uv_unwrap_mesh_test : public ::testing::Test
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
TEST(uv_unwrap_mesh, create)
{
  using namespace kwiver::vital;

  plugin_manager::instance().load_all_plugins();

  EXPECT_NE(nullptr, algo::uv_unwrap_mesh::create("core"));
}

// ----------------------------------------------------------------------------
TEST_F(uv_unwrap_mesh_test, check_texture_coordinates)
{
  uv_unwrap_mesh mesh_unwrap;
  config_block_sptr algo_config = mesh_unwrap.get_configuration();
  algo_config->set_value<double>("spacing", 0.005);
  mesh_unwrap.set_configuration(algo_config);

  mesh_unwrap.unwrap(mesh);

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
