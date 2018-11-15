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
 * \brief Operations to modify meshes
 */

#ifndef KWIVER_ARROWS_CORE_MESH_OPERATIONS_H
#define KWIVER_ARROWS_CORE_MESH_OPERATIONS_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/mesh.h>


namespace kwiver {
namespace arrows {
namespace core {


/// Subdivide mesh faces into triangles
/**
 * \param [in]  faces  An array of generic mesh faces to triangulate
 * \returns     An array of triangles covering the input faces
 */
KWIVER_ALGO_CORE_EXPORT
std::unique_ptr<kwiver::vital::mesh_regular_face_array<3> >
mesh_triangulate(const kwiver::vital::mesh_face_array_base& faces);


/// Subdivide quadrilaterals into triangles
/**
 * \param [in]  faces  An array of quad mesh faces to triangulate
 * \returns     An array of triangles covering the input faces
 */
KWIVER_ALGO_CORE_EXPORT
std::unique_ptr<kwiver::vital::mesh_regular_face_array<3> >
mesh_triangulate(const kwiver::vital::mesh_regular_face_array<4>& faces);


/// Triangulate the faces of the mesh (in place)
/**
 * \param [in,out]  mesh  A mesh to triangulate faces in place
 */
KWIVER_ALGO_CORE_EXPORT
void
mesh_triangulate(kwiver::vital::mesh& mesh);


}
}
}
#endif // KWIVER_ARROWS_CORE_MESH_OPERATIONS_H
