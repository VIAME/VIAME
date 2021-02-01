// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for a map from frame IDs to vpgl cameras
 */

#ifndef KWIVER_ARROWS_VXL_CAMERA_MAP_H_
#define KWIVER_ARROWS_VXL_CAMERA_MAP_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/camera_map.h>

#include <vpgl/vpgl_perspective_camera.h>

#include <map>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A concrete camera_map that wraps a map of vpgl_perspective_camera
class KWIVER_ALGO_VXL_EXPORT camera_map
: public vital::camera_map
{
public:
  /// typedef for a map of frame numbers to vpgl_perspective_camera
  typedef std::map<kwiver::vital::frame_id_t, vpgl_perspective_camera<double> > map_vcam_t;

  /// Default Constructor
  camera_map() {}

  /// Constructor from a std::map of vpgl_perspective_camera
  explicit camera_map(const map_vcam_t& cameras)
  : data_(cameras) {}

  /// Return the number of cameras in the map
  virtual size_t size() const { return data_.size(); }

  /// Return a map from integer IDs to camera shared pointers
  virtual map_camera_t cameras() const;

  /// Return underlying map from IDs to vpgl_perspective_camera
  virtual map_vcam_t vpgl_cameras() const { return data_; }

protected:

  /// The map from integer IDs to vpgl_perspective_camera
  map_vcam_t data_;
};

/// Convert any camera map to a vpgl camera map
KWIVER_ALGO_VXL_EXPORT
camera_map::map_vcam_t
camera_map_to_vpgl(const vital::camera_map& cam_map);

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
