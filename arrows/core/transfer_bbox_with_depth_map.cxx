// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "transfer_bbox_with_depth_map.h"

#include <math.h>
#include <assert.h>
#include <tuple>
#include <iostream>
#include <sstream>
#include <vital/algo/image_io.h>
#include <vital/io/camera_io.h>
#include <vital/config/config_difference.h>
#include <vital/util/string.h>
#include <Eigen/Core>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

// ---------------------------------------------------------------------------
int
nearest_index(int max, double value)
{
  if (abs(value - 0.0) < 1e-6)
  {
    return 0;
  }
  else if (abs(value - (double)max) < 1e-6)
  {
    return max - 1;
  }
  else
  {
    return (int)round(value - 0.5);
  }
}

// ---------------------------------------------------------------------------
vector_3d
backproject_to_depth_map
(kwiver::vital::camera_perspective_sptr const camera,
 kwiver::vital::image_container_sptr const depth_map,
 vector_2d const& img_pt)
{
  vector_2d npt_ = camera->intrinsics()->unmap(img_pt);
  auto npt = vector_3d(npt_(0), npt_(1), 1.0);

  matrix_3x3d M = camera->rotation().matrix().transpose();
  vector_3d cam_pos = camera->center();

  vector_3d Mp = M * npt;

  kwiver::vital::image dm_data = depth_map->get_image();
  auto dm_width = (int)dm_data.width();
  auto dm_height = (int)dm_data.height();

  int img_pt_x = nearest_index(dm_width, img_pt(0));
  int img_pt_y = nearest_index(dm_height, img_pt(1));

  if (img_pt_x < 0 || img_pt_y < 0 ||
      img_pt_x >= dm_width ||
      img_pt_y >= dm_height)
  {
    throw std::invalid_argument("Provided image point is outside of image "
                                "bounds");
  }

  float depth = dm_data.at<float>(img_pt_x, img_pt_y);

  vector_3d world_pos = cam_pos + (Mp * depth);

  return world_pos;
}

// ---------------------------------------------------------------------------
std::tuple<vector_3d, vector_3d>
backproject_wrt_height
(kwiver::vital::camera_perspective_sptr const camera,
 kwiver::vital::image_container_sptr const depth_map,
 vector_2d const& img_pt_bottom,
 vector_2d const& img_pt_top)
{
  vector_3d world_pos_bottom = backproject_to_depth_map
    (camera, depth_map, img_pt_bottom);

  vector_2d npt_ = camera->intrinsics()->unmap(img_pt_top);
  auto npt = vector_3d(npt_(0), npt_(1), 1.0);

  matrix_3x3d M = camera->rotation().matrix().transpose();
  vector_3d cam_pos = camera->center();

  vector_3d Mp = M * npt;

  double xf = world_pos_bottom(0);
  double yf = world_pos_bottom(1);

  double xc = cam_pos(0);
  double yc = cam_pos(1);
  double zc = cam_pos(2);

  double nx = Mp(0);
  double ny = Mp(1);
  double nz = Mp(2);

  // If we assume that the top world point for given pair is directly
  // above the bottom world point at (xf, yf, zf), then the top world
  // point is at (xf, yf, zh), where we need to solve for zh.  The
  // camera is located at (xc, yc, zc) and the ray coming out at the
  // top image point is along the direction (in world coordinates)
  // <nx, ny, nz>, but we don't know how far along this ray's
  // direction we need to go to be as close as possible to (xf, yf,
  // zf). If we travel 't' along the ray, then the squared distance
  // from (xf, yf) is (xc+nx*t - xf)^2 + (yc+ny*t - yf)^2, and we want
  // 't' that minimizes this. Take the derivative wrt 't' and set
  // equal to zero:
  // 2 * (xc + nx * t - xf) * nx + 2 * (yc + ny * t - yf) * ny = 0
  // Rearranged as:
  double t = (ny * (yf - yc) + nx * (xf - xc)) /
    (std::pow(nx, 2) + std::pow(ny, 2));

  double zh = zc + t*nz;

  auto world_pos_top = vector_3d(xf, yf, zh);

  return std::tuple<vector_3d, vector_3d> (world_pos_bottom, world_pos_top);
}

// ---------------------------------------------------------------------------
vital::bounding_box<double>
transfer_bbox_with_depth_map_stationary_camera
(kwiver::vital::camera_perspective_sptr const src_camera,
 kwiver::vital::camera_perspective_sptr const dest_camera,
 kwiver::vital::image_container_sptr const depth_map,
 vital::bounding_box<double> const bbox)
{
  double bbox_min_x = bbox.min_x();
  double bbox_max_x = bbox.max_x();
  double bbox_min_y = bbox.min_y();
  double bbox_max_y = bbox.max_y();
  double bbox_aspect_ratio = (bbox_max_x - bbox_min_x) /
    (bbox_max_y - bbox_min_y);

  auto bbox_bottom_center = vector_2d
    ((bbox_max_x + bbox_min_x) / 2, bbox_max_y);
  auto bbox_top_center = vector_2d
    ((bbox_max_x + bbox_min_x) / 2, bbox_min_y);

  vector_3d world_pos_bottom;
  vector_3d world_pos_top;

  std::tie(world_pos_bottom, world_pos_top) =
    backproject_wrt_height
    (src_camera, depth_map, bbox_bottom_center, bbox_top_center);

  vector_2d dest_img_pos_bottom = dest_camera->project(world_pos_bottom);
  vector_2d dest_img_pos_top = dest_camera->project(world_pos_top);

  double dest_bbox_min_y = dest_img_pos_top(1);
  double dest_bbox_max_y = dest_img_pos_bottom(1);
  double dest_bbox_height = dest_bbox_max_y - dest_bbox_min_y;

  // Using the original bbox aspect ratio to compute the width of our
  // transferred box.  Could use a more sophisticated method here
  double dest_bbox_width_d = bbox_aspect_ratio * dest_bbox_height;

  // Use the average center x coordinate of transfered top and bottom
  // points as the center of the bounding box
  double dest_bbox_min_x =
    ((dest_img_pos_top(0) + dest_img_pos_bottom(0)) / 2) -
    (dest_bbox_width_d / 2);
  double dest_bbox_max_x =
    ((dest_img_pos_top(0) + dest_img_pos_bottom(0)) / 2) +
    (dest_bbox_width_d / 2);

  return vital::bounding_box<double>
    (dest_bbox_min_x, dest_bbox_min_y, dest_bbox_max_x, dest_bbox_max_y);
}

// ---------------------------------------------------------------------------
transfer_bbox_with_depth_map::
transfer_bbox_with_depth_map()
{
}

// ---------------------------------------------------------------------------
transfer_bbox_with_depth_map::
transfer_bbox_with_depth_map
(kwiver::vital::camera_perspective_sptr src_cam,
 kwiver::vital::camera_perspective_sptr dest_cam,
 kwiver::vital::image_container_sptr src_cam_depth_map)
  : src_camera( src_cam )
  , dest_camera( dest_cam )
  , depth_map( src_cam_depth_map )
{
}

// ---------------------------------------------------------------------------
vital::config_block_sptr
transfer_bbox_with_depth_map::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "src_camera_krtd_file_name", src_camera_krtd_file_name,
                     "Source camera KRTD file name path" );

  config->set_value( "dest_camera_krtd_file_name", dest_camera_krtd_file_name,
                     "Destination camera KRTD file name path" );

  config->set_value( "src_camera_depth_map_file_name",
                     src_camera_depth_map_file_name,
                     "Source camera depth map file name path" );

  vital::algo::image_io::
    get_nested_algo_configuration( "image_reader", config, image_reader );

  return config;
}

// ---------------------------------------------------------------------------
void
transfer_bbox_with_depth_map::
set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );
  this->src_camera_krtd_file_name =
    config->get_value< std::string > ( "src_camera_krtd_file_name" );
  this->dest_camera_krtd_file_name =
    config->get_value< std::string > ( "dest_camera_krtd_file_name" );
  this->src_camera_depth_map_file_name =
    config->get_value< std::string > ( "src_camera_depth_map_file_name" );

  // Setup actual reader algorithm
  vital::algo::image_io::
    set_nested_algo_configuration( "image_reader", config, image_reader );

  this->src_camera =
    kwiver::vital::read_krtd_file( this->src_camera_krtd_file_name );
  this->dest_camera =
    kwiver::vital::read_krtd_file( this->dest_camera_krtd_file_name );

  this->depth_map = image_reader->load( this->src_camera_depth_map_file_name );
}

// ---------------------------------------------------------------------------
bool
transfer_bbox_with_depth_map::
check_configuration( vital::config_block_sptr config ) const
{
  kwiver::vital::config_difference cd( this->get_configuration(), config );
  const auto key_list = cd.extra_keys();

  if ( ! key_list.empty() )
  {
    LOG_WARN( logger(), "Additional parameters found in config block that are "
                        "not required or desired: "
                        << kwiver::vital::join( key_list, ", " ) );
  }

  return true;
}

// ---------------------------------------------------------------------------
vital::detected_object_set_sptr
transfer_bbox_with_depth_map::
filter( vital::detected_object_set_sptr const input_set ) const
{
  auto ret_set = std::make_shared<vital::detected_object_set>();

  for ( auto det : *input_set )
  {
    auto out_det = det->clone();
    auto out_bbox = out_det->bounding_box();

    try
    {
      vital::bounding_box<double> new_out_bbox =
        transfer_bbox_with_depth_map_stationary_camera
        (src_camera, dest_camera, depth_map, out_bbox);
      out_det->set_bounding_box( new_out_bbox );
      ret_set->add( out_det );
    }
    catch (const std::invalid_argument& e)
    {
      std::ostringstream strs;
      strs << "Bounding box ("
           << out_bbox.min_x() << ", "
           << out_bbox.min_y() << ", "
           << out_bbox.max_x() << ", "
           << out_bbox.max_y() << ") "
           << "couldn't be transferred, skipping!";

      LOG_WARN(logger(), strs.str());
    }
  }

  return ret_set;
}

}}} // end namespace
