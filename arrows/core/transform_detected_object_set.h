// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_
#define KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_filter.h>
#include <vital/io/camera_io.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

/// Transforms detections based on source and destination cameras.
class KWIVER_ALGO_CORE_EXPORT transform_detected_object_set
  : public vital::algo::detected_object_filter
{
public:
  PLUGIN_INFO( "transform_detected_object_set",
               "Transforms a detected object set based on source and "
               "destination cameras.\n\n" )

  /// Default constructor
  transform_detected_object_set();

  /// Constructor taking source and destination cameras directly
  transform_detected_object_set(kwiver::vital::camera_perspective_sptr src_cam,
                                kwiver::vital::camera_perspective_sptr dest_cam);

  /// Default destructor
  virtual ~transform_detected_object_set() = default;

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

  kwiver::vital::camera_perspective_sptr src_camera;
  kwiver::vital::camera_perspective_sptr dest_camera;

  virtual vital::bounding_box<double>
    transform_bounding_box(vital::bounding_box<double> const& bbox) const;

  virtual vector_3d
    backproject_to_ground(kwiver::vital::camera_perspective_sptr const camera,
                          vector_2d const& img_pt) const;

  virtual vector_3d
    backproject_to_plane(kwiver::vital::camera_perspective_sptr const camera,
                         vector_2d const& img_pt,
                         vector_4d const& plane) const;

  virtual Eigen::Matrix<double, 8, 3>
    backproject_bbox(kwiver::vital::camera_perspective_sptr const camera,
                     vital::bounding_box<double> const& bbox) const;

  virtual vital::bounding_box<double>
    box_around_box3d(kwiver::vital::camera_perspective_sptr const camera,
                     Eigen::Matrix<double, 8, 3> const& box3d) const;

  virtual vital::bounding_box<double>
    view_to_view(kwiver::vital::camera_perspective_sptr const src_camera,
                 kwiver::vital::camera_perspective_sptr const dest_camera,
                 vital::bounding_box<double> const& bbox) const;
};

}}} //End namespace

#endif // KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_
