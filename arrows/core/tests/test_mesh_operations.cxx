// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <gtest/gtest.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/core/mesh_operations.h>

#include <vital/types/mesh.h>
#include <vital/io/mesh_io.h>
#include <vital/io/camera_io.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
kwiver::vital::mesh_sptr generate_mesh()
{
  std::unique_ptr<mesh_vertex_array_base>
    verts(new mesh_vertex_array<3>(
  {
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1}
  }));

  std::unique_ptr<mesh_regular_face_array<4>>
    faces(new mesh_regular_face_array<4>(
  {
    {0, 2, 3, 1},
    {0, 4, 6, 2},
    {5, 7, 6, 4},
    {1, 3, 7, 5},
    {2, 6, 7, 3},
    {4, 0, 1, 5}
  }));

  return std::make_shared<mesh>(std::move(verts), std::move(faces));
}

// ----------------------------------------------------------------------------
kwiver::vital::mesh_sptr generate_mesh_plane(double s=1)
{
  std::unique_ptr<mesh_vertex_array_base>
    verts(new mesh_vertex_array<3>(
  {
    { -s, -s, 0 },
    { -s,  s, 0 },
    {  s,  s, 0 },
    {  s, -s, 0 }
  }));

  std::unique_ptr<mesh_regular_face_array<4>>
    faces(new mesh_regular_face_array<4>(
  {
    { 0, 1, 2, 3 }
  }));

  return std::make_shared<mesh>(std::move(verts), std::move(faces));
}

// ----------------------------------------------------------------------------
TEST(mesh_operations, triangulate_mesh)
{
  // Mesh
  mesh_sptr mesh = generate_mesh();

  EXPECT_EQ( mesh->faces().regularity(), 4 );
  EXPECT_EQ( mesh->num_faces(), 6 );

  kwiver::arrows::core::mesh_triangulate(*mesh);

  EXPECT_EQ( mesh->faces().regularity(), 3 );
  EXPECT_EQ( mesh->num_faces(), 12 );
}

// ----------------------------------------------------------------------------
TEST(mesh_operations, clip_mesh)
{
  // No clipping case
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: full mesh preserved" << std::endl;
    kwiver::vital::vector_4d plane(1, 0, 0, 2);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_FALSE(clipped);
    EXPECT_EQ(mesh->num_faces(), 12);
    EXPECT_EQ(mesh->num_verts(), 8);
    kwiver::vital::write_obj("test_noclip.obj", *mesh);
  }

  // Clip everything case
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: full mesh clipped" << std::endl;
    kwiver::vital::vector_4d plane(1, 0, 0, -2);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 0);
    EXPECT_EQ(mesh->num_verts(), 8);
    kwiver::vital::write_obj("test_allclip.obj", *mesh);
  }

  // Partial clipping
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: half mesh clipped" << std::endl;
    kwiver::vital::vector_4d plane(1, 0, 0, 0);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 14);
    EXPECT_EQ(mesh->num_verts(), 16);

    kwiver::vital::write_obj("test_clip.obj", *mesh);
  }

  // Partial clipping - vertex on the plane
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: half mesh clipped, vert on plane" << std::endl;
    kwiver::vital::vector_4d plane(1, 1, 0, 0);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 8);
    EXPECT_EQ(mesh->num_verts(), 10);

    kwiver::vital::write_obj("test_clip_vert.obj", *mesh);
  }

  // Partial clipping - edge on the plane
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: half mesh clipped, edge on plane" << std::endl;
    kwiver::vital::vector_4d plane(1, 1, 0, 2);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 12);
    EXPECT_EQ(mesh->num_verts(), 8);

    kwiver::vital::write_obj("test_clip_edge.obj", *mesh);
  }

  // Partial clipping - face on the plane
  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);

    std::cout << "Clipping case: half mesh clipped, face on plane" << std::endl;
    kwiver::vital::vector_4d plane(1, 0, 0, 1);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, plane);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 10);
    EXPECT_EQ(mesh->num_verts(), 8);

    kwiver::vital::write_obj("test_clip_face.obj", *mesh);
  }
}

// ----------------------------------------------------------------------------
TEST(mesh_operations, clip_mesh_camera)
{
  // generate test camera
  kwiver::vital::simple_camera_perspective camera;
  camera.set_center({ 2, 0, 2 });
  camera.look_at({ 0, 0, 0 });
  kwiver::vital::simple_camera_intrinsics K(600.0, { 320, 240 });
  K.set_image_width(640);
  K.set_image_height(480);
  camera.set_intrinsics(K.clone());
  kwiver::vital::write_krtd_file(camera, "test_clip_cam.krtd");

  {
    mesh_sptr mesh = generate_mesh();
    kwiver::arrows::core::mesh_triangulate(*mesh);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh, camera, 1.5, 4.0);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh->num_faces(), 51);
    EXPECT_EQ(mesh->num_verts(), 58);
    kwiver::vital::write_obj("test_clip_cam.obj", *mesh);
  }

  {
    camera.set_center({ 2, 2, 2 });
    camera.look_at({ 0, 0, 0 });
    kwiver::vital::write_krtd_file(camera, "test_clip_cam_plane.krtd");
    mesh_sptr mesh_plane = generate_mesh_plane(10);
    kwiver::arrows::core::mesh_triangulate(*mesh_plane);
    bool clipped = kwiver::arrows::core::clip_mesh(*mesh_plane, camera);

    EXPECT_TRUE(clipped);
    EXPECT_EQ(mesh_plane->num_faces(), 6);
    EXPECT_EQ(mesh_plane->num_verts(), 16);
    kwiver::vital::write_obj("test_clip_cam_plane.obj", *mesh_plane);
  }
}
