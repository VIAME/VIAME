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


// The immediately following copyright notice applies to code in the methods
// Triangulate_DLT, find_optimal_image_points and triangulate_fast_two_view.

 // Copyright (C) 2013 The Regents of the University of California (Regents).
 // All rights reserved.
 //
 // Redistribution and use in source and binary forms, with or without
 // modification, are permitted provided that the following conditions are
 // met:
 //
 //     * Redistributions of source code must retain the above copyright
 //       notice, this list of conditions and the following disclaimer.
 //
 //     * Redistributions in binary form must reproduce the above
 //       copyright notice, this list of conditions and the following
 //       disclaimer in the documentation and/or other materials provided
 //       with the distribution.
 //
 //     * Neither the name of The Regents or University of California nor the
 //       names of its contributors may be used to endorse or promote products
 //       derived from this software without specific prior written permission.
 //
 // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 // ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 // LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 // CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 // SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 // INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 // CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 // ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 // POSSIBILITY OF SUCH DAMAGE.
 //
 // Please contact the author of this library if you have any questions.
 // Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)


/**
 * \file
 * \brief Implementation of triangulation function
 */

#include "triangulate.h"
#include <Eigen/SVD>
#include <arrows/core/epipolar_geometry.h>


namespace kwiver {
namespace arrows {

// Triangulates 2 views
void
Triangulate_DLT( kwiver::vital::matrix_3x4d const& pose1,
                 kwiver::vital::matrix_3x4d const& pose2,
                 kwiver::vital::vector_2d const& point1,
                 kwiver::vital::vector_2d const& point2,
                 kwiver::vital::vector_4d &triangulated_point)
{
  // code modified from code found at
  // https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/triangulation/triangulation.h

  kwiver::vital::matrix_4x4d design_matrix;
  design_matrix.row(0) = point1[0] * pose1.row(2) - pose1.row(0);
  design_matrix.row(1) = point1[1] * pose1.row(2) - pose1.row(1);
  design_matrix.row(2) = point2[0] * pose2.row(2) - pose2.row(0);
  design_matrix.row(3) = point2[1] * pose2.row(2) - pose2.row(1);

  // Extract nullspace.
  Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(design_matrix, Eigen::ComputeFullV);
  triangulated_point = svd.matrixV().rightCols<1>();
}

// Given either a fundamental or essential matrix and two corresponding images
// points such that ematrix * point2 produces a line in the first image,
// this method finds corrected image points such that
// corrected_point1^t * ematrix * corrected_point2 = 0.
void
find_optimal_image_points(kwiver::vital::essential_matrix_sptr ematrix,
                          const vital::vector_2d &point1,
                          const vital::vector_2d &point2,
                          vital::vector_2d &corrected_point1,
                          vital::vector_2d &corrected_point2)
{
  // code modified from code found at
  // https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/triangulation/triangulation.cc
  auto E = ematrix->matrix();

  vital::vector_3d point1_homog = point1.homogeneous();
  vital::vector_3d point2_homog = point2.homogeneous();

  // A helper matrix to isolate certain coordinates.
  Eigen::Matrix<double, 2, 3> s_matrix;
  s_matrix << 1, 0, 0, 0, 1, 0;

  const Eigen::Matrix2d e_submatrix = E.topLeftCorner<2, 2>();

  // The epipolar line from one image point in the other image.
  vital::vector_2d epipolar_line1 = s_matrix * E * point2_homog;
  vital::vector_2d epipolar_line2 = s_matrix * E.transpose() * point1_homog;

  const double a = epipolar_line1.transpose() * e_submatrix * epipolar_line2;
  const double b =
    (epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm()) / 2.0;
  const double c = point1_homog.transpose() * E * point2_homog;

  const double d = sqrt(b * b - a * c);

  double lambda = c / (b + d);
  epipolar_line1 -= e_submatrix * lambda * epipolar_line1;
  epipolar_line2 -= e_submatrix.transpose() * lambda * epipolar_line2;

  lambda *=
    (2.0 * d) / (epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm());

  corrected_point1 =
    (point1_homog - s_matrix.transpose() * lambda * epipolar_line1)
    .hnormalized();
  corrected_point2 =
    (point2_homog - s_matrix.transpose() * lambda * epipolar_line2)
    .hnormalized();
}

template <typename T>
Eigen::Matrix<T, 3, 1>
triangulate_fast_two_view(const vital::simple_camera_perspective &camera0,
                          const vital::simple_camera_perspective &camera1,
                          const Eigen::Matrix<T, 2, 1> &point0,
                          const Eigen::Matrix<T, 2, 1> &point1)
{
  // code modified from code found at
  // https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/triangulation/triangulation.cc

  auto E = kwiver::arrows::essential_matrix_from_cameras(camera0, camera1);

  const vital::vector_2d pt0 = camera0.get_intrinsics()->unmap(point0.template cast<double>());
  const vital::vector_2d pt1 = camera1.get_intrinsics()->unmap(point1.template cast<double>());

  vital::vector_2d corrected_pt0, corrected_pt1;

  find_optimal_image_points(E, pt0, pt1, corrected_pt0, corrected_pt1);

  vital::vector_4d triangulated_point;
  Triangulate_DLT(camera0.pose_matrix(), camera1.pose_matrix(), corrected_pt0, corrected_pt1, triangulated_point);
  return triangulated_point.hnormalized().template cast<T>();
}

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
template KWIVER_ALGO_CORE_EXPORT Eigen::Matrix<T,3,1> \
         triangulate_fast_two_view( \
            const vital::simple_camera_perspective &camera0, \
            const vital::simple_camera_perspective &camera1, \
            const Eigen::Matrix<T, 2, 1> &point0, \
            const Eigen::Matrix<T, 2, 1> &point1); \
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
