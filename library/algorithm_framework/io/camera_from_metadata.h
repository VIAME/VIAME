// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Function to generate \ref kwiver::vital::camera_rpc from metadata

#ifndef VITAL_CAMERA_FROM_METADATA_H_
#define VITAL_CAMERA_FROM_METADATA_H_

#include <viame/algorithm_framework/vital_export.h>

#include <viame/core_types/camera_intrinsics.h>
#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/camera_rpc.h>
#include <viame/core_types/local_tangent_space.h>
#include <viame/core_types/metadata.h>

namespace kwiver {

namespace vital {

/// Convert space separated sting to Eigen vector
///
/// \param s The string to be converted.
/// \return The converted vector.
VITAL_EXPORT
vector_d
string_to_vector( std::string const& s );

/// Produce RPC camera from metadata
///
/// \param file_path   The path to the file to read in.
/// \return A new camera object representing the contents of the read-in file.
VITAL_EXPORT
camera_sptr
camera_from_metadata( metadata_sptr const& md );

/// Use metadata to construct intrinsics
///
/// \param [in]  md            A metadata object to extract intrinsics from
/// \param [in]  image_width   The width of the image
/// \param [in]  image_height  The height of the image
/// \returns nullptr if insufficient data to construct intrinsics
VITAL_EXPORT
camera_intrinsics_sptr
intrinsics_from_metadata(
  metadata const& md,
  size_t image_width,
  size_t image_height );

/// Use a sequence of metadata objects to initialize a sequence of cameras.
///
/// \param [in] md_map A mapping from frame number to metadata object.
/// \param [in] base_camera The camera to reposition at each metadata pose.
/// \param [in,out] local_space
///  The local cartesian coordinate system used for the cameras.
/// \param [in] init_intrinsics
///   Initialize intrinsics with metadata.  If set false then use the
///   base_camera intrinsics.
/// \param [in] rot_offset
///   Rotation offset to apply to yaw/pitch/roll metadata before updating a
///   camera's rotation.
/// \returns A mapping from frame number to camera.
/// \note
///   The \c local_space object is updated only if it is not already valid.
///   If updated, the computed local origin is determined from the mean camera
///   position at zero altitude.
VITAL_EXPORT
std::map< frame_id_t, camera_sptr >
initialize_cameras_with_metadata(
  std::map< frame_id_t, metadata_sptr > const& md_map,
  simple_camera_perspective const& base_camera,
  local_tangent_space& local_space,
  bool init_intrinsics = true,
  rotation_d const& rot_offset = rotation_d() );

/// Use the pose data provided by metadata to update camera pose.
///
/// \param metadata
///   The metadata packet to update the camera with.
/// \param cam The camera to be updated.
/// \param rot_offset
///   A rotation offset to apply to metadata rotation data.
///
/// \return \c true if metadata is sufficient to update the camera.
VITAL_EXPORT
bool
update_camera_from_metadata(
  metadata const& md,
  local_tangent_space const& local_space,
  simple_camera_perspective& cam,
  rotation_d const& rot_offset = rotation_d() );

/// Update a sequence of metadata from a sequence of cameras.
///
/// \param [in] cam_map A mapping from frame number to camera.
/// \param [in] local_space
///  The local cartesian coordinate system used for the cameras.
/// \param [in,out] md_map
///   A mapping from frame_number of metadata objects to update. If no
///   metadata object is found for a frame, a new one is created.
VITAL_EXPORT
void
update_metadata_from_cameras(
  std::map< frame_id_t, camera_sptr > const& cam_map,
  local_tangent_space const& local_space,
  std::map< frame_id_t, metadata_sptr >& md_map );

/// Use the camera pose to update the metadata structure.
///
/// \param [in] cam The camera data.
/// \param [in] local_space
///  The local cartesian coordinate system used for the camera.
/// \param [in,out] md The metadata object to update in place.
VITAL_EXPORT
void
update_metadata_from_camera(
  simple_camera_perspective const& cam,
  local_tangent_space const& local_space,
  metadata& md );

} // namespace vital

} // namespace kwiver

#endif
