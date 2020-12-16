// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Operations to modify meshes
 */

#ifndef KWIVER_ARROWS_CORE_MESH_OPERATIONS_H
#define KWIVER_ARROWS_CORE_MESH_OPERATIONS_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/mesh.h>
#include <vital/types/camera_perspective.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Subdivide mesh faces into triangles
/**
 * \param [in]  faces  An array of generic mesh faces to triangulate
 * \returns     An array of triangles covering the input faces
 *
 * \note This implementation assumes that each face is convex and does not
 *       consider mesh geometry when deciding how to best split faces.
 */
KWIVER_ALGO_CORE_EXPORT
std::unique_ptr<kwiver::vital::mesh_regular_face_array<3> >
mesh_triangulate(kwiver::vital::mesh_face_array_base const& faces);

/// Subdivide quadrilaterals into triangles
/**
 * \param [in]  faces  An array of quad mesh faces to triangulate
 * \returns     An array of triangles covering the input faces
 *
 * \note This implementation assumes that each face is convex and does not
 *       consider mesh geometry when deciding how to best split faces.
 */
KWIVER_ALGO_CORE_EXPORT
std::unique_ptr<kwiver::vital::mesh_regular_face_array<3> >
mesh_triangulate(kwiver::vital::mesh_regular_face_array<4> const& faces);

/// Triangulate the faces of the mesh (in place)
/**
 * \param [in,out]  mesh  A mesh to triangulate faces in place
 *
 * \note This implementation assumes that each face is convex and does not
 *       consider mesh geometry when deciding how to best split faces.
 */
KWIVER_ALGO_CORE_EXPORT
void
mesh_triangulate(kwiver::vital::mesh& mesh);

/// Clip a triangular mesh with a plane
/**
 * Intersect a mesh with a plane and keep only the parts of the mesh that
 * lie on the positive side of the plane (direction that the normal points).
 * Faces crossing the plane are intersected with the plane and new
 * vertices are added along the plane.  This implementation does not remove
 * or renumber existing vertices, but may add new vertices.  It may leave
 * vertices which are no longer used by the faces.
 *
 * \param [in,out]  mesh  A mesh to triangulate clip in place
 * \param [in]      plane The clipping plane
 *
 * \note This implementation assumes that the mesh is triangular
 */
KWIVER_ALGO_CORE_EXPORT
bool
clip_mesh(kwiver::vital::mesh& mesh,
          kwiver::vital::vector_4d const& plane);

/// Clip a triangular mesh with a camera frustum
/**
 * Intersect a mesh with a camera frustum and keep only the parts of the mesh
 * that lie inside the frustum.  Faces crossing the frustum planes are
 * intersected with the planes and new vertices are added along the planes.
 * This implementation does not remove or renumber existing vertices, but may
 * add new vertices.  It may leave vertices which are no longer used by the
 * faces.
 *
 * \param [in,out]  mesh   A mesh to triangulate clip in place
 * \param [in]      camera The camera frustum to use in clipping
 * \param [in]      near   The offset from the camera center to the near
 *                         clipping plane (parallel to the image plane)
 * \param [in]      far    The offset from the camera center to the far
 *                         clipping plane (parallel to the image plane)
 * \param [in]      margin Expand the frustum by this many pixels on all sides
 *                         to avoid boundary effects from clipping too tightly.
 *
 * By default this function keeps all geometry in front of the camera that
 * would project into the image.  The far clipping plane is set to infinity
 * to disable far plane clipping.
 *
 * \note This implementation assumes that the mesh is triangular
 */
KWIVER_ALGO_CORE_EXPORT
bool
clip_mesh(kwiver::vital::mesh& mesh,
          kwiver::vital::camera_perspective const& camera,
          double near = 0.0,
          double far = std::numeric_limits<double>::infinity(),
          double margin = 1.0);

}
}
}
#endif // KWIVER_ARROWS_CORE_MESH_OPERATIONS_H
