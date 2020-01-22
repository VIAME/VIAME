/*ckwg +29
 * Copyright 2014-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation of Necker reversal functions
 */

#include "necker_reverse.h"
#include <Eigen/Geometry>
#include <Eigen/SVD>


namespace kwiver {
namespace arrows {
namespace core {


/// Compute a plane passing through the landmarks
vital::vector_4d
landmark_plane(const vital::landmark_map::map_landmark_t& landmarks)
{
  using namespace kwiver::vital;
  // compute the landmark location mean and covariance
  vector_3d lc(0.0, 0.0, 0.0);
  matrix_3x3d covar = matrix_3x3d::Zero();
  for (auto const& p : landmarks)
  {
    vector_3d pt = p.second->loc();
    lc += pt;
    covar += pt * pt.transpose();
  }
  const double num_lm = static_cast<double>(landmarks.size());
  lc /= num_lm;
  covar /= num_lm;
  covar -= lc * lc.transpose();

  // the plane will pass through the landmark centeroid (lc)
  // and have a normal vector aligned with the smallest eigenvector of covar
  Eigen::JacobiSVD<matrix_3x3d> svd(covar, Eigen::ComputeFullV);
  vector_3d axis = svd.matrixV().col(2);
  return vector_4d(axis.x(), axis.y(), axis.z(), -(lc.dot(axis)));
}


/// Mirror landmarks about the specified plane
vital::landmark_map_sptr
mirror_landmarks(vital::landmark_map const& landmarks,
                 vital::vector_4d const& plane)
{
  using namespace kwiver::vital;
  landmark_map::map_landmark_t new_lms;
  const vector_3d axis(plane.x(), plane.y(), plane.z());
  const double d = plane[3];
  // mirror landmark locations about the mirroring plane
  for (auto const& p : landmarks.landmarks())
  {
    vector_3d v = p.second->loc();
    v -= 2.0 * (v.dot(axis) + d) * axis;
    auto new_lm = std::make_shared<vital::landmark_d>(*p.second);
    new_lm->set_loc(v);
    new_lms[p.first] = new_lm;
  }
  return std::make_shared<simple_landmark_map>(new_lms);
}


/// Compute the Necker reversal of a camera in place
void
necker_reverse_inplace(vital::simple_camera_perspective& camera,
                       vital::vector_4d const& plane)
{
  using namespace kwiver::vital;
  const vector_3d axis(plane.x(), plane.y(), plane.z());
  const double d = plane[3];
  const rotation_d Ra180(vector_4d(axis.x(), axis.y(), axis.z(), 0.0));
  static const rotation_d Rz180(vector_4d(0.0, 0.0, 1.0, 0.0));

  // extract the camera center
  const vital::vector_3d cc = camera.center();
  // extract the camera principal axis
  const vital::vector_3d pa = camera.rotation().matrix().row(2);
  // compute the distance from cc along pa until intersection with
  // the mirroring plane of the points
  const double dist = -(cc.dot(axis) + d) / pa.dot(axis);
  // compute the ground point where the principal axis
  // intersects the mirroring plane
  const vital::vector_3d gp = cc + dist * pa;
  // rotate the camera center 180 degrees about the mirroring plane normal
  // axis centered at gp, also rotate the camera 180 about its principal axis
  camera.set_center(Ra180 * (cc - gp) + gp);
  camera.set_rotation(Rz180 * camera.rotation() * Ra180);
}


/// Compute the Necker reversal of the cameras
vital::camera_map_sptr
necker_reverse(vital::camera_map const& cameras,
               vital::vector_4d const& plane)
{
  using namespace kwiver::vital;
  camera_map::map_camera_t cams;
  // flip cameras around
  for (auto & p : cameras.cameras())
  {
    // make a clone of this camera as a simple_camera_perspective
    auto flipped = std::make_shared<vital::simple_camera_perspective>(
      dynamic_cast<vital::simple_camera_perspective&>(*p.second));

    necker_reverse_inplace(*flipped, plane);
    cams[p.first] = flipped;
  }
  return std::make_shared<simple_camera_map>(cams);
}


/// Compute an approximate Necker reversal of cameras and landmarks
void
necker_reverse(vital::camera_map_sptr& cameras,
               vital::landmark_map_sptr& landmarks,
               bool reverse_landmarks)
{
  using namespace kwiver::vital;
  const vector_4d plane = landmark_plane(landmarks->landmarks());

  // reverse the cameras
  cameras = necker_reverse(*cameras, plane);

  if (reverse_landmarks)
  {
    // mirror the landmarks
    landmarks = mirror_landmarks(*landmarks, plane);
  }
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
