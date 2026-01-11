// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Header file for loading camera collections and stereo rigs

#ifndef VIAME_CAMERA_RIG_IO_H_
#define VIAME_CAMERA_RIG_IO_H_

#include <plugins/core/viame_core_export.h>

#include <vital/types/camera_rig.h>
#include <vital/vital_types.h>

#include <vector>

namespace viame {

/// Load a camera rig from KRtd file(s).
///
/// \throws invalid_data
///   Unable to find any camera files in the given directory
/// \throw path_not_exists
///   The specified directory does not exist
///
/// \param cam_files a list of camera names
/// \return a new camera rig
kwiver::vital::camera_rig_sptr
VIAME_CORE_EXPORT read_camera_rig( kwiver::vital::path_list_t const& cam_files );

/// Load a stereo rig from a file or directory.
///
/// Supports the following formats:
/// - .json: JSON format with camera intrinsics and extrinsics
/// - .yml/.yaml: OpenCV YAML format with camera matrices
/// - .npz: NumPy compressed archive (requires ZLIB)
/// - directory: OpenCV calibration directory with intrinsics.yml and extrinsics.yml
///
/// \throws invalid_data
///   Unable to find any camera files in the given directory
/// \throw path_not_exists
///   The specified directory does not exist
///
/// \param FN input file name or directory path
/// \return a new stereo rig
kwiver::vital::camera_rig_stereo_sptr
VIAME_CORE_EXPORT read_stereo_rig( kwiver::vital::path_t const& FN );

/// Load a stereo rig from a JSON file.
///
/// JSON format expects fields: fx_left, fy_left, cx_left, cy_left,
/// k1_left, k2_left, p1_left, p2_left, k3_left (and same for right),
/// plus R (rotation matrix) and T (translation vector).
///
/// \param FN path to the JSON file
/// \return a new stereo rig
kwiver::vital::camera_rig_stereo_sptr
VIAME_CORE_EXPORT read_stereo_rig_json( kwiver::vital::path_t const& FN );

/// Load a stereo rig from an OpenCV YAML file.
///
/// YAML format expects matrices M1, D1, M2, D2 (camera intrinsics and
/// distortion coefficients) and R, T (rotation and translation).
///
/// \param FN path to the YAML file
/// \return a new stereo rig
kwiver::vital::camera_rig_stereo_sptr
VIAME_CORE_EXPORT read_stereo_rig_yaml( kwiver::vital::path_t const& FN );

/// Load a stereo rig from an OpenCV calibration directory.
///
/// Expects a directory containing intrinsics.yml and extrinsics.yml files
/// in OpenCV FileStorage format with matrices M1, D1, M2, D2, R, T.
///
/// \param dir_path path to the calibration directory
/// \return a new stereo rig
kwiver::vital::camera_rig_stereo_sptr
VIAME_CORE_EXPORT read_stereo_rig_from_ocv_dir( kwiver::vital::path_t const& dir_path );

#ifdef VIAME_ENABLE_ZLIB
/// Load a stereo rig from a NumPy NPZ file.
///
/// NPZ format expects arrays: R, T, cameraMatrixL, cameraMatrixR,
/// and optionally distCoeffsL, distCoeffsR.
///
/// \param FN path to the NPZ file
/// \return a new stereo rig
kwiver::vital::camera_rig_stereo_sptr
VIAME_CORE_EXPORT read_stereo_rig_npz( kwiver::vital::path_t const& FN );
#endif

/// Save a camera rig to KRtd file(s)
///
/// \throws invalid_data
///   Unable to find any camera files in the given directory
/// \throw path_not_exists
///   The specified directory does not exist
///
/// \param rig camera rig
void
VIAME_CORE_EXPORT write_camera_rig( kwiver::vital::camera_rig_sptr rig );

/// Save a stereo rig to a file.
///
/// Supported formats: .json
///
/// \throws invalid_data
///   Unable to find any camera files in the given directory
/// \throw path_not_exists
///   The specified directory does not exist
///
/// \param rig stereo rig
/// \param FN output file name
void
VIAME_CORE_EXPORT write_stereo_rig( kwiver::vital::camera_rig_stereo_sptr rig,
                                     std::string const& FN );

} // namespace viame

#endif // VIAME_CAMERA_RIG_IO_H_
