// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for MVG triangulation function
 */

#ifndef KWIVER_ARROWS_MVG_TRIANGULATE_H_
#define KWIVER_ARROWS_MVG_TRIANGULATE_H_

#include <vital/vital_config.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/camera_rpc.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Triangulate a 3D point from a set of cameras and 2D image points
/**
 *  This function computes a linear least squares solution find a 3D point
 *  that is the closest intersection of all the rays using an inhomogeneous
 *  system of equations.  This method is affine invariant but does not work
 *  for 3D points at infinity.
 *
 *  \param cameras a vector of camera objects
 *  \param points a vector of image points corresponding to each camera
 *  \return a 3D triangulated point location
 */
template <typename T>
KWIVER_ALGO_MVG_EXPORT
Eigen::Matrix<T,3,1>
triangulate_inhomog(const std::vector<vital::simple_camera_perspective >& cameras,
                    const std::vector<Eigen::Matrix<T,2,1> >& points);

/// Triangulate a 3D point from a set of cameras and 2D image points
/**
* This function uses only the first two cameras and two points to triangulate.
* It uses the method laid out in the paper "Triangulation Made Easy" Lindstrom
* CVPR 2010.  This approach is efficient and offers an alternative approach that
* may be numerically superior in some cases.  This approach does not work for
* points at infinty.
*
*  \param camera0 the first camera
*  \param camera1 the second camera
*  \param point0 a 2d point in the first camera
*  \param point1 a matching 2d point in the second camera
*  \return a 3D triangulated point location
*/
template <typename T>
KWIVER_ALGO_MVG_EXPORT
Eigen::Matrix<T, 3, 1>
triangulate_fast_two_view(const vital::simple_camera_perspective &camera0,
                          const vital::simple_camera_perspective &camera1,
                          const Eigen::Matrix<T, 2, 1> &point0,
                          const Eigen::Matrix<T, 2, 1> &point1);

/// Triangulate a homogeneous 3D point from a set of cameras and 2D image points
/**
 *  This function computes a linear least squares solution find a homogeneous
 *  3D point that is the closest intersection of all the rays using a
 *  homogeneous system of equations.  This method is not invariant to
 *  tranformations but does allow for 3D points at infinity.
 *
 *  \param cameras a vector of camera objects
 *  \param points a vector of image points corresponding to each camera
 *  \return a homogeneous 3D triangulated point location
 */
template <typename T>
KWIVER_ALGO_MVG_EXPORT
Eigen::Matrix<T,4,1>
triangulate_homog(const std::vector<vital::simple_camera_perspective >& cameras,
                  const std::vector<Eigen::Matrix<T,2,1> >& points);

/// Triangulate a 3D point from a set of RPC cameras and 2D image points
/**
 *  This function constructs rays at two arbitary heights using the cameras and
 *  image points. Then a least squares solution is used to find the 3D point
 *
 *  \param cameras a vector of RPC camera objects
 *  \param points a vector of image points corresponding to each camera
 *  \return a 3D triangulated point location
 */
template <typename T>
KWIVER_ALGO_MVG_EXPORT
Eigen::Matrix<T,3,1>
triangulate_rpc(const std::vector<vital::simple_camera_rpc >& cameras,
                const std::vector<Eigen::Matrix<T,2,1> >& points);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
