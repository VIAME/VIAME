// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_TRANSFER_BBOX_WITH_DEPTH_MAP_H_
#define KWIVER_ARROWS_TRANSFER_BBOX_WITH_DEPTH_MAP_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_filter.h>
#include <vital/io/camera_io.h>
#include <vital/algo/image_io.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

/// Backproject image point to a depth map
KWIVER_ALGO_CORE_EXPORT
vector_3d
  backproject_to_depth_map
  (kwiver::vital::camera_perspective_sptr const camera,
   kwiver::vital::image_container_sptr const depth_map,
   vector_2d const& img_pt);

/// Backproject an image point (top) assumed to be directly above another
KWIVER_ALGO_CORE_EXPORT
std::tuple<vector_3d, vector_3d>
  backproject_wrt_height
  (kwiver::vital::camera_perspective_sptr const camera,
   kwiver::vital::image_container_sptr const depth_map,
   vector_2d const& img_pt_bottom,
   vector_2d const& img_pt_top);

/// Transfer a bounding box wrt two cameras and a depth map
KWIVER_ALGO_CORE_EXPORT
vital::bounding_box<double>
  transfer_bbox_with_depth_map_stationary_camera
  (kwiver::vital::camera_perspective_sptr const src_camera,
   kwiver::vital::camera_perspective_sptr const dest_camera,
   kwiver::vital::image_container_sptr const depth_map,
   vital::bounding_box<double> const bbox);

/// Transforms detections based on source and destination cameras.
class KWIVER_ALGO_CORE_EXPORT transfer_bbox_with_depth_map
  : public vital::algo::detected_object_filter
{
public:
  PLUGIN_INFO( "transfer_bbox_with_depth_map",
               "Transforms detected object set bounding boxes based on source "
               "and destination cameras with respect the source cameras depth "
               "map.\n\n" )

  /// Default constructor
  transfer_bbox_with_depth_map();

  /// Constructor taking source and destination cameras directly
  transfer_bbox_with_depth_map
    (kwiver::vital::camera_perspective_sptr src_cam,
     kwiver::vital::camera_perspective_sptr dest_cam,
     kwiver::vital::image_container_sptr src_cam_depth_map);

  /// Get this algorithm's configuration block
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Apply the transformation
  virtual vital::detected_object_set_sptr
    filter( vital::detected_object_set_sptr const input_set) const;

private:
  std::string src_camera_krtd_file_name;
  std::string dest_camera_krtd_file_name;
  std::string src_camera_depth_map_file_name;

  std::shared_ptr<vital::algo::image_io> image_reader;

  kwiver::vital::camera_perspective_sptr src_camera;
  kwiver::vital::camera_perspective_sptr dest_camera;
  kwiver::vital::image_container_sptr depth_map;
};

}}} //End namespace

#endif // KWIVER_ARROWS_TRANSFER_BBOX_WITH_DEPTH_MAP_H_
