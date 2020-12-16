// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for 3D transformation functions
 */

#ifndef KWIVER_ARROWS_MVG_TRANSFORM_H_
#define KWIVER_ARROWS_MVG_TRANSFORM_H_

#include <vital/vital_config.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/camera_perspective_map.h>
#include <vital/types/similarity.h>
#include <vital/types/covariance.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Transform a 3D covariance matrix with a similarity transformation
/**
 *  This function applies the scale and rotation of a similarity transformation
 *  to a covariance matrix such that the Mahalanobis distance measure between
 *  two points remains unchanged after applying the same transformation to the
 *  points.  That is,
 *       (x1-m1)'*C1*(x1-m1) == (x2-m2)'*C2*(x2-m2)
 *       for x2 = xform*x1 and m2 = xform*m1 and C2 = transform(C1, xform)
 *
 *  \param [in] covar the 3D covariance to transform
 *  \param [in] xform the 3D similarity transformation to apply
 *  \return a 3D covariance transformed by the similarity transformation
 */
template <typename T>
KWIVER_ALGO_MVG_EXPORT
vital::covariance_<3,T> transform(const vital::covariance_<3,T>& covar,
                                  const vital::similarity_<T>& xform);

/// Transform the camera by applying a similarity transformation in place
KWIVER_ALGO_MVG_EXPORT
void transform_inplace(vital::simple_camera_perspective& cam,
                       const vital::similarity_d& xform);

/// Transform the camera map by applying a similarity transformation in place
KWIVER_ALGO_MVG_EXPORT
void transform_inplace(vital::simple_camera_perspective_map& cameras,
                       const vital::similarity_d& xform);

/// Transform the landmark by applying a similarity transformation in place
template <typename T>
KWIVER_ALGO_MVG_EXPORT
void transform_inplace(vital::landmark_<T>& lm,
                       const vital::similarity_<T>& xform);

/// Transform the landmark map by applying a similarity transformation in place
KWIVER_ALGO_MVG_EXPORT
void transform_inplace(vital::landmark_map& landmarks,
                       const vital::similarity_d& xform);

/// Transform the landmark map by applying a similarity transformation in place
KWIVER_ALGO_MVG_EXPORT
void transform_inplace(vital::landmark_map::map_landmark_t& landmarks,
                       const vital::similarity_d& xform);

/// construct a transformed camera by applying a similarity transformation
KWIVER_ALGO_MVG_EXPORT
vital::camera_perspective_sptr transform(vital::camera_perspective_sptr cam,
                                         const vital::similarity_d& xform);

/// construct a transformed map of cameras by applying a similarity transformation
KWIVER_ALGO_MVG_EXPORT
vital::camera_map_sptr transform(vital::camera_map_sptr cameras,
                                 const vital::similarity_d& xform);

/// construct a transformed map of cameras by applying a similarity transformation
KWIVER_ALGO_MVG_EXPORT
vital::camera_perspective_map_sptr
transform(vital::camera_perspective_map_sptr cameras,
          const vital::similarity_d& xform);

/// construct a transformed landmark by applying a similarity transformation
KWIVER_ALGO_MVG_EXPORT
vital::landmark_sptr transform(vital::landmark_sptr lm,
                               const vital::similarity_d& xform);

/// construct a transformed map of landmarks by applying a similarity transformation
KWIVER_ALGO_MVG_EXPORT
vital::landmark_map_sptr transform(vital::landmark_map_sptr landmarks,
                                   const vital::similarity_d& xform);

/// translate landmarks in place by the provided offset vector
KWIVER_ALGO_MVG_EXPORT
void translate_inplace(vital::landmark_map& landmarks,
                       vital::vector_3d const& offset);

/// translate cameras in place by the provided offset vector
KWIVER_ALGO_MVG_EXPORT
void translate_inplace(vital::simple_camera_perspective_map& cameras,
                       vital::vector_3d const& offset);

/// translate cameras in place by the provided offset vector
/**
 * \note only translates cameras which are perspective and have a
 * defined center
 */
KWIVER_ALGO_MVG_EXPORT
void translate_inplace(vital::camera_map& cameras,
                       vital::vector_3d const& offset);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
