/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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
 * \brief Implementation of \link kwiver::vital::camera_rpc
 * camera_rpc \endlink class
 */

#include <vital/types/camera_rpc.h>
#include <vital/io/eigen_io.h>
#include <vital/types/matrix.h>
#include <Eigen/Geometry>

#include <iomanip>

namespace kwiver {
namespace vital {


camera_rpc
::camera_rpc()
  : m_logger( kwiver::vital::get_logger( "vital.camera_rpc" ) )
{
}



/// Project a 3D point into a 2D image point
vector_2d
camera_rpc
::project( const vector_3d& pt ) const
{
  // Normalize points
  vector_3d norm_pt =
    ( pt - world_offset() ).cwiseQuotient( world_scale() );

  // Calculate polynomials
  // TODO: why doesn't this work ?
  // auto polys = this->rpc_coeffs()*this->power_vector(norm_pt);
  auto rpc = this->rpc_coeffs();
  auto pv = this->power_vector(norm_pt);
  auto polys = rpc*pv;
  vector_2d image_pt( polys[0] / polys[1], polys[2] / polys[3]);

  // Un-normalize
  return image_pt.cwiseProduct( image_scale() ) + image_offset();
}

Eigen::Matrix<double, 20, 1>
simple_camera_rpc
::power_vector( const vector_3d& pt ) const
{
  // Form the monomials in homogeneous form
  double w  = 1.0;
  double x = pt.x();
  double y = pt.y();
  double z = pt.z();

  double xx = x * x;
  double xy = x * y;
  double xz = x * z;
  double yy = y * y;
  double yz = y * z;
  double zz = z * z;
  double xxx = xx * x;
  double xxy = xx * y;
  double xxz = xx * z;
  double xyy = xy * y;
  double xyz = xy * z;
  double xzz = xz * z;
  double yyy = yy * y;
  double yyz = yy * z;
  double yzz = yz * z;
  double zzz = zz * z;

  // Fill in vector
  Eigen::Matrix<double, 20, 1> retVec;
  retVec << w, x, y, z, xy, xz, yz, xx, yy, zz,
            xyz, xxx, xyy, xzz, xxy, yyy, yzz, xxz, yyz, zzz;
  return retVec;
}

} } // end namespace
