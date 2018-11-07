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

/**
 * \file
 * \brief Implementation of kwiver::arrows::render_mesh_depth_map function
 */

#ifndef KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H
#define KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/image_container.h>
#include <vital/types/mesh.h>


namespace kwiver {
namespace arrows {


/// This function renders a depthmap of a mesh seen by a camera
/**
 * @brief render_mesh_depth_map
 * @param mesh [in]
 * @param camera [in]
 * @return a depth map
 */
KWIVER_ALGO_CORE_EXPORT
vital::image_container_sptr render_mesh_depth_map(kwiver::vital::mesh_sptr mesh,
                                                  kwiver::vital::camera_perspective_sptr camera);


/// This function fills image with depth interpolated
/**
 * @brief rasterize_depth
 * @param mesh [in]
 * @param image_points [in] image coordinates of vertices
 * @param depth [in]
 * @param image [out]
 * @param check_partial_intersection [in] the pixels which partially intersect
 * with a triangle can be assigned to it.
 */
KWIVER_ALGO_CORE_EXPORT
void rasterize_depth(const vital::mesh_regular_face_array<3> &triangles,
                     const std::vector<vital::vector_2d> &image_points,
                     const std::vector<double> &depth, vital::image_of<double> &image,
                     bool check_partial_intersection=false);

}
}
#endif // KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H
