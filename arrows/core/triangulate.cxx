/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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
 * \brief Implementation of triangulation function
 */

#include "triangulate.h"
#include <Eigen/SVD>


namespace kwiver {
namespace arrows {

/// Triangulate a 3D point from a set of cameras and 2D image points
template <typename T>
Eigen::Matrix<T,3,1>
triangulate_inhomog(const std::vector<vital::simple_camera_perspective >& cameras,
                    const std::vector<Eigen::Matrix<T,2,1> >& points)
{
  typedef Eigen::Matrix<T,2,1> vector_2;
  typedef Eigen::Matrix<T,3,1> vector_3;
  typedef Eigen::Matrix<T,3,3> matrix_3x3;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 3> data_matrix_t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> data_vector_t;
  const unsigned int num_rows = 2*static_cast<unsigned int>(points.size());
  data_matrix_t A(num_rows, 3);
  data_vector_t b(num_rows);
  for ( unsigned int i=0; i<points.size(); ++i )
  {
    // the camera
    const vital::simple_camera_perspective& cam = cameras[i];
    const matrix_3x3 R(cam.get_rotation().matrix().cast<T>());
    const vector_3 t(cam.translation().cast<T>());
    // the point in normalized coordinates
    const vital::vector_2d p2d = points[i].template cast<double>();
    const vector_2 pt = cam.get_intrinsics()->unmap(p2d).template cast<T>();
    A(2*i,   0) = R(0,0) - pt.x() * R(2,0);
    A(2*i,   1) = R(0,1) - pt.x() * R(2,1);
    A(2*i,   2) = R(0,2) - pt.x() * R(2,2);
    A(2*i+1, 0) = R(1,0) - pt.y() * R(2,0);
    A(2*i+1, 1) = R(1,1) - pt.y() * R(2,1);
    A(2*i+1, 2) = R(1,2) - pt.y() * R(2,2);
    b[2*i  ] = t.z()*pt.x() - t.x();
    b[2*i+1] = t.z()*pt.y() - t.y();
  }
  Eigen::JacobiSVD<data_matrix_t> svd(A, Eigen::ComputeFullU |
                                         Eigen::ComputeFullV);
  return svd.solve(b);
}


/// Triangulate a homogeneous 3D point from a set of cameras and 2D image points
template <typename T>
Eigen::Matrix<T,4,1>
triangulate_homog(const std::vector<vital::simple_camera_perspective >& cameras,
                  const std::vector<Eigen::Matrix<T,2,1> >& points)
{
  typedef Eigen::Matrix<T,2,1> vector_2;
  typedef Eigen::Matrix<T,3,1> vector_3;
  typedef Eigen::Matrix<T,3,3> matrix_3x3;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 4> data_matrix_t;
  const unsigned int num_rows = 2*static_cast<unsigned int>(points.size());
  data_matrix_t A(num_rows, 4);
  for ( unsigned int i=0; i<points.size(); ++i )
  {
    // the camera
    const vital::simple_camera_perspective& cam = cameras[i];
    const matrix_3x3 R(cam.get_rotation().matrix().cast<T>());
    const vector_3 t(cam.translation().cast<T>());
    // the point in normalized coordinates
    const vital::vector_2d p2d = points[i].template cast<double>();
    const vector_2 pt = cam.get_intrinsics()->unmap(p2d).template cast<T>();
    A(2*i,   0) = R(0,0) - pt.x() * R(2,0);
    A(2*i,   1) = R(0,1) - pt.x() * R(2,1);
    A(2*i,   2) = R(0,2) - pt.x() * R(2,2);
    A(2*i,   3) = t.x()  - pt.x() * t.z();
    A(2*i+1, 0) = R(1,0) - pt.y() * R(2,0);
    A(2*i+1, 1) = R(1,1) - pt.y() * R(2,1);
    A(2*i+1, 2) = R(1,2) - pt.y() * R(2,2);
    A(2*i+1, 3) = t.y()  - pt.y() * t.z();
  }
  Eigen::JacobiSVD<data_matrix_t > svd(A, Eigen::ComputeFullV);
  return svd.matrixV().col(3);
}


/// Triangulate a 3D point from a set of RPC cameras and 2D image points
template <typename T>
Eigen::Matrix<T,3,1>
triangulate_rpc(const std::vector<vital::simple_camera_rpc >& cameras,
                const std::vector<Eigen::Matrix<T,2,1> >& points)
{
  // Get the pairs of points to define the rays
  std::vector< std::pair< vital::vector_3d, vital::vector_3d > > pts;

  Eigen::Array3d curr_scale = cameras[0].world_scale().array();
  Eigen::Array3d curr_offset = cameras[0].world_offset().array();
  Eigen::Array3d min_pos = curr_offset - curr_scale;
  Eigen::Array3d max_pos = curr_offset + curr_scale;

  for ( unsigned int i = 0; i < points.size(); ++i )
  {
    // Get world offset and scale to set normalization and sample heights
    curr_scale = cameras[i].world_scale().array();
    curr_offset = cameras[i].world_offset().array();

    min_pos = min_pos.min( curr_offset - curr_scale );
    max_pos = max_pos.max( curr_offset + curr_scale );

    double h1 = ( curr_offset - curr_scale )[2];
    double h2 = ( curr_offset + curr_scale )[2];

    vital::vector_3d pt1 =
      cameras[i].back_project( points[i].template cast<double>(), h1 );
    vital::vector_3d pt2 =
      cameras[i].back_project( points[i].template cast<double>(), h2 );
    pts.push_back(
      std::pair< vital::vector_3d, vital::vector_3d >( pt1, pt2 ) );
  }

  // Get normalization factors for full point set
  vital::vector_3d scale = 0.5*( max_pos.matrix() - min_pos.matrix() );
  vital::vector_3d offset = 0.5*( max_pos.matrix() + min_pos.matrix() );

  vital::matrix_3x3d M = vital::matrix_3x3d::Zero();
  vital::vector_3d v(0., 0., 0.);

  for ( auto& pt : pts )
  {
    // Normalize points
    vital::vector_3d p = ( pt.first - offset ).cwiseQuotient( scale );
    vital::vector_3d x = ( pt.second - offset ).cwiseQuotient( scale );

    // Unit vector along ray
    vital::vector_3d unit_vec = ( x - p ).normalized();

    vital::matrix_3x3d tmp_mat =
      vital::matrix_3x3d::Identity() - unit_vec * unit_vec.transpose();
    M += tmp_mat;
    v += tmp_mat * p;
  }

  // Un-normalize before return
  return ( scale.cwiseProduct(
    M.colPivHouseholderQr().solve( v ) ) + offset ).cast<T>();
}

/// \cond DoxygenSuppress
#define INSTANTIATE_TRIANGULATE(T) \
template KWIVER_ALGO_CORE_EXPORT Eigen::Matrix<T,4,1> \
         triangulate_homog( \
            const std::vector<vital::simple_camera_perspective >& cameras, \
            const std::vector<Eigen::Matrix<T,2,1> >& points); \
template KWIVER_ALGO_CORE_EXPORT Eigen::Matrix<T,3,1> \
         triangulate_inhomog( \
            const std::vector<vital::simple_camera_perspective >& cameras, \
            const std::vector<Eigen::Matrix<T,2,1> >& points); \
template KWIVER_ALGO_CORE_EXPORT Eigen::Matrix<T,3,1> \
         triangulate_rpc( \
            const std::vector<vital::simple_camera_rpc >& cameras, \
            const std::vector<Eigen::Matrix<T,2,1> >& points);

INSTANTIATE_TRIANGULATE(double);
INSTANTIATE_TRIANGULATE(float);

#undef INSTANTIATE_TRIANGULATE
/// \endcond


} // end namespace arrows
} // end namespace kwiver
