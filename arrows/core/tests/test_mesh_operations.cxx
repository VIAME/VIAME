/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include <arrows/core/mesh_operations.h>

#include <vital/types/mesh.h>


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
