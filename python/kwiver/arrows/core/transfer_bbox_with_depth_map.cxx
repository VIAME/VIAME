// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
