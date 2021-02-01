// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header defining VXL algorithm implementation of camera optimization.
*/

#include "optimize_cameras.h"

#include <map>
#include <vector>
#include <utility>

#include <vital/exceptions.h>

#include <arrows/vxl/camera.h>
#include <arrows/vxl/camera_map.h>

#include <vgl/vgl_homg_point_3d.h>
#include <vgl/vgl_point_2d.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_double_3.h>
#include <vpgl/algo/vpgl_optimize_camera.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

namespace // anonymous
{

/// Reproduction of vpgl_optimize_camera::opt_orient_pos(...), but without
/// trace statement.
vpgl_perspective_camera<double>
opt_orient_pos(vpgl_perspective_camera<double> const& camera,
               std::vector<vgl_homg_point_3d<double> > const& world_points,
               std::vector<vgl_point_2d<double> > const& image_points)
{
  const vpgl_calibration_matrix<double>& K = camera.get_calibration();
  vgl_point_3d<double> c = camera.get_camera_center();
  const vgl_rotation_3d<double>& R = camera.get_rotation();

  // compute the Rodrigues vector from the rotation
  vnl_double_3 w = R.as_rodrigues();

  vpgl_orientation_position_lsqr lsqr_func(K,world_points,image_points);
  vnl_levenberg_marquardt lm(lsqr_func);
  vnl_vector<double> params(6);
  params[0]=w[0];   params[1]=w[1];   params[2]=w[2];
  params[3]=c.x();  params[4]=c.y();  params[5]=c.z();
  lm.minimize(params);
  vnl_double_3 w_min(params[0],params[1],params[2]);
  vgl_homg_point_3d<double> c_min(params[3], params[4], params[5]);

  return vpgl_perspective_camera<double>(K, c_min, vgl_rotation_3d<double>(w_min) );
}

}

/// Optimize a single camera given corresponding features and landmarks
void
optimize_cameras
::optimize(vital::camera_perspective_sptr& camera,
           const std::vector<vital::feature_sptr>& features,
           const std::vector<vital::landmark_sptr>& landmarks,
           kwiver::vital::sfm_constraints_sptr constraints) const
{
  if( constraints && constraints->get_metadata()->size() != 0 )
  {
    LOG_WARN( vital::get_logger( "arrows.vxl.optimize_cameras" ),
              "constraints are provided but will be ignored by this algorithm");
  }

  // remove camera intrinsics from the camera and work in normalized coordinates
  // VXL is only optimizing rotation and translation and doesn't model distortion
  vital::simple_camera_perspective mcamera(*camera);
  vital::camera_intrinsics_sptr k(camera->intrinsics());
  mcamera.set_intrinsics(vital::camera_intrinsics_sptr(new vital::simple_camera_intrinsics()));

  // convert the camera
  vpgl_perspective_camera<double> vcamera;
  vital_to_vpgl_camera(mcamera, vcamera);

  // For each camera in the input map, create corresponding point sets for 2D
  // and 3D coordinates of tracks and matching landmarks, respectively, for
  // that camera's frame.
  std::vector< vgl_point_2d<double> > pts_2d;
  std::vector< vgl_homg_point_3d<double> > pts_3d;
  vector_2d tmp_2d;
  vector_3d tmp_3d;
  for( unsigned int i=0; i<features.size(); ++i )
  {
    // unmap the points to normalized coordinates
    tmp_2d = k->unmap(features[i]->loc());
    tmp_3d = landmarks[i]->loc();
    pts_2d.push_back(vgl_point_2d<double>(tmp_2d.x(), tmp_2d.y()));
    pts_3d.push_back(vgl_homg_point_3d<double>(tmp_3d.x(), tmp_3d.y(), tmp_3d.z()));
  }
  // optimize
  vcamera = opt_orient_pos(vcamera, pts_3d, pts_2d);

  // convert back and fill in the unchanged intrinsics
  vpgl_camera_to_vital(vcamera, mcamera);
  mcamera.set_intrinsics(k);
  camera = std::dynamic_pointer_cast<vital::camera_perspective>(mcamera.clone());
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
