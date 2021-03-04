// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Typedefs for camera_perspective camera_maps
 */

#ifndef VITAL_CAMERA_PERSPECTIVE_MAP_H_
#define VITAL_CAMERA_PERSPECTVIE_MAP_H_

#include "camera_map.h"
#include "camera_perspective.h"

namespace kwiver {
namespace vital {

/// Type aliases that combine camera_map_of_ and camera_perspective
using camera_perspective_map =
        vital::camera_map_of_<vital::camera_perspective>;
using camera_perspective_map_sptr =
        std::shared_ptr<camera_perspective_map>;

/// Type aliases that combine camera_map_of_ and simple_camera_perspective
using simple_camera_perspective_map =
        vital::camera_map_of_<vital::simple_camera_perspective>;
using simple_camera_perspective_map_sptr =
        std::shared_ptr<simple_camera_perspective_map>;

} // end namespace vital
} // end namespace kwiver

#endif
