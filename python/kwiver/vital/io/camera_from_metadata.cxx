// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/io/camera_from_metadata.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( camera_from_metadata, m )
{
  m.doc() = "Python bindings for KWIVER camera-from-metadata functions";

  m.def(
    "intrinsics_from_metadata",
    &kv::intrinsics_from_metadata,
    py::arg( "md" ),
    py::arg( "image_width" ),
    py::arg( "image_height" ),
    R"doc(
    Create camera intrinsics from metadata.

    Args:
        md: Metadata containing sensor parameters
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        SimpleCameraIntrinsics or None if insufficient metadata
    )doc" );

  m.def(
    "update_camera_from_metadata",
    &kv::update_camera_from_metadata,
    py::arg( "md" ),
    py::arg( "local_space" ),
    py::arg( "cam" ),
    py::arg( "rot_offset" ) = kv::rotation_d(),
    R"doc(
    Update camera pose from metadata.

    Args:
        md: Metadata containing position and orientation
        local_space: Local geographic coordinate system
        cam: Camera to update (modified in-place)
        rot_offset: Optional rotation offset

    Returns:
        True if camera was successfully updated
    )doc" );

  m.def(
    "initialize_cameras_with_metadata",
    &kv::initialize_cameras_with_metadata,
    py::arg( "md_map" ),
    py::arg( "base_camera" ),
    py::arg( "local_space" ),
    py::arg( "init_intrinsics" ) = true,
    py::arg( "rot_offset" ) = kv::rotation_d(),
    R"doc(
    Initialize cameras from metadata map.

    Creates a camera for each frame with valid metadata. Sets local
    geographic coordinate system origin based on camera positions.

    Args:
        md_map: Map of frame_id to metadata
        base_camera: Base camera with default intrinsics
        local_space: Local geographic coordinate system (updated with origin)
        init_intrinsics: Whether to initialize intrinsics from metadata
        rot_offset: Optional rotation offset

    Returns:
        Map of frame_id to camera
    )doc" );
}
