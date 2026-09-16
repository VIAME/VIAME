// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "transfer_bbox_with_depth_map.h"

#include <arrows/core/transfer_bbox_with_depth_map.h>
#include <viame/core_types/camera.h>
#include <viame/core_types/image.h>
#include <viame/core_types/image_container.h>

void
transfer_bbox_with_depth_map( py::module& m )
{
  m.def(
    "backproject_to_depth_map",
    [](std::shared_ptr< viame::simple_camera_perspective > const src_cam,
       std::shared_ptr< viame::simple_image_container > const img_cont,
       viame::vector_2d const& img_pt){
      return viame::core::
             backproject_to_depth_map( src_cam, img_cont, img_pt );
    } );

  m.def(
    "backproject_wrt_height",
    [](std::shared_ptr< viame::simple_camera_perspective > const src_cam,
       std::shared_ptr< viame::simple_image_container > const img_cont,
       viame::vector_2d const& img_pt_bottom,
       viame::vector_2d const& img_pt_top){
      return viame::core::
             backproject_wrt_height(
        src_cam, img_cont, img_pt_bottom,
        img_pt_top );
    } );

  m.def(
    "transfer_bbox_with_depth_map_stationary_camera",
    [](std::shared_ptr< viame::simple_camera_perspective > const src_cam,
       std::shared_ptr< viame::simple_camera_perspective > const
       dest_cam,
       std::shared_ptr< viame::simple_image_container > const img_cont,
       viame::bounding_box< double > const bbox){
      return viame::core::
             transfer_bbox_with_depth_map_stationary_camera(
        src_cam, dest_cam,
        img_cont, bbox );
    } );
}
