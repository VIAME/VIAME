/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#include "transfer_bbox_with_depth_map.h"

#include <arrows/core/transfer_bbox_with_depth_map.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/types/camera.h>

void transfer_bbox_with_depth_map(py::module &m)
{
  m.def("backproject_to_depth_map",
        [](std::shared_ptr<kwiver::vital::simple_camera_perspective> const src_cam,
           std::shared_ptr<kwiver::vital::simple_image_container> const img_cont,
           kwiver::vital::vector_2d const& img_pt)
        {
          return kwiver::arrows::core::
            backproject_to_depth_map(src_cam, img_cont, img_pt);
        });

  m.def("backproject_wrt_height",
        [](std::shared_ptr<kwiver::vital::simple_camera_perspective> const src_cam,
           std::shared_ptr<kwiver::vital::simple_image_container> const img_cont,
           kwiver::vital::vector_2d const& img_pt_bottom,
           kwiver::vital::vector_2d const& img_pt_top)
        {
          return kwiver::arrows::core::
            backproject_wrt_height(src_cam, img_cont, img_pt_bottom, img_pt_top);
        });

  m.def("transfer_bbox_with_depth_map_stationary_camera",
        [](std::shared_ptr<kwiver::vital::simple_camera_perspective> const src_cam,
           std::shared_ptr<kwiver::vital::simple_camera_perspective> const dest_cam,
           std::shared_ptr<kwiver::vital::simple_image_container> const img_cont,
           kwiver::vital::bounding_box<double> const bbox)
        {
          return kwiver::arrows::core::
            transfer_bbox_with_depth_map_stationary_camera
            (src_cam, dest_cam, img_cont, bbox);
        });
}
