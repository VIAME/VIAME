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

/**
 * \file
 * \brief Implementation of mesh operations
 */

#include "mesh_operations.h"



namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;

/// Subdivide mesh faces into triangle
std::unique_ptr<mesh_regular_face_array<3> >
mesh_triangulate(const mesh_face_array_base& faces)
{
  std::unique_ptr<mesh_regular_face_array<3> > tris(new mesh_regular_face_array<3>);
  int group = -1;
  if (faces.has_groups())
    group = 0;
  for (unsigned int f=0; f<faces.size(); ++f) {
    for (unsigned i=2; i<faces.num_verts(f); ++i)
      tris->push_back(mesh_tri(faces(f,0),faces(f,i-1),faces(f,i)));
    if (group >= 0 && f+1 >= faces.groups()[group].second) {
      tris->make_group(faces.groups()[group++].first);
    }
  }
  return tris;
}


/// Subdivide quadrilaterals into triangle
std::unique_ptr<mesh_regular_face_array<3> >
mesh_triangulate(const mesh_regular_face_array<4>& faces)
{
  std::unique_ptr<mesh_regular_face_array<3> > tris(new mesh_regular_face_array<3>);
  int group = -1;
  if (faces.has_groups())
    group = 0;
  for (unsigned int f=0; f<faces.size(); ++f) {
    const mesh_regular_face<4>& face = faces[f];
    tris->push_back(mesh_tri(face[0],face[1],face[2]));
    tris->push_back(mesh_tri(face[0],face[2],face[3]));
    if (group >= 0 && f+1 >= faces.groups()[group].second) {
      tris->make_group(faces.groups()[group++].first);
    }
  }
  return tris;
}


/// Triangulate the faces of the mesh (in place)
void
mesh_triangulate(mesh& mesh)
{
  switch (mesh.faces().regularity())
  {
    case 3: break;
    case 4:
    {
      std::unique_ptr<mesh_face_array_base> tris(
          mesh_triangulate(static_cast<const mesh_regular_face_array<4>&>(mesh.faces())));
      mesh.set_faces(std::move(tris));
      break;
    }
    default:
    {
      std::unique_ptr<mesh_face_array_base> tris(mesh_triangulate(mesh.faces()));
      mesh.set_faces(std::move(tris));
      break;
    }
  }
}

}
}
}
