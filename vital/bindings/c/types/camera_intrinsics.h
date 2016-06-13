/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief C Interface to \p vital::camera_intrinsics class
 */

#ifndef VITAL_C_CAMERA_INTRINSICS_H_
#define VITAL_C_CAMERA_INTRINSICS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/types/eigen.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>


/// Opaque pointer to camera intrinsics objects
typedef struct vital_camera_intrinsics_s vital_camera_intrinsics_t;


/// Create new simple camera intrinsics object with default parameters
/**
 * \param eh Vital error handle structure
 * \returns New camera intrinsics object
 */
VITAL_C_EXPORT
vital_camera_intrinsics_t*
vital_camera_intrinsics_new_default( vital_error_handle_t *eh );


/**
 * Create a new simple camera intrinsics object with specified focal length and
 * principle point.
 *
 * \param focal_length The focal length
 * \param principle_point The 2D principle point
 * \param eh Vital error handle structure
 * \returns New camera intrinsics object
 */
VITAL_C_EXPORT
vital_camera_intrinsics_t*
vital_camera_intrinsics_new_partial( double focal_length,
                                     vital_eigen_matrix2x1d_t *principle_point,
                                     vital_error_handle_t *eh );


/// Create a new simple camera intrinsics object
/**
 * \param focal_length The focal length
 * \param principle_point The 2D principle point
 * \param aspect_ratio The aspect ratio
 * \param skew The skew
 * \param dist_coeffs_data Pointer to an array of doubles comprising existing
 *                         distortion coefficients. This may be empty by
 *                         specifying a zero length in the next parameter.
 * \param dist_coeffs_size The number of coefficient values in the
 *                         ``dist_coeffs_data`` array. This may be zero to
 *                         specify no array.
 * \param eh Vital error handle structure
 * \returns New camera intrinsics object
 */
VITAL_C_EXPORT
vital_camera_intrinsics_t*
vital_camera_intrinsics_new( double focal_length,
                             vital_eigen_matrix2x1d_t *principle_point,
                             double aspect_ratio,
                             double skew,
                             vital_eigen_matrixXx1d_t *dist_coeffs,
                             vital_error_handle_t *eh );


/// Destroy a given non-null camera intrinsics object
VITAL_C_EXPORT
void
vital_camera_intrinsics_destroy( vital_camera_intrinsics_t *ci,
                                 vital_error_handle_t *eh );


/// Get the focal length
/**
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns Double valued focal length
 */
VITAL_C_EXPORT
double
vital_camera_intrinsics_get_focal_length( vital_camera_intrinsics_t *ci,
                                          vital_error_handle_t *eh );


/// Get a new copy of the principle point
/**
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns 2 by 1 matrix that is the principle point.
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_get_principle_point( vital_camera_intrinsics_t *ci,
                                             vital_error_handle_t *eh );


/// Get the aspect ratio
/**
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns Double valued aspect ratio
 */
VITAL_C_EXPORT
double
vital_camera_intrinsics_get_aspect_ratio( vital_camera_intrinsics_t *ci,
                                          vital_error_handle_t *eh );


/// Get the skew value
/**
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns Double valued skew.
 */
VITAL_C_EXPORT
double
vital_camera_intrinsics_get_skew( vital_camera_intrinsics_t *ci,
                                  vital_error_handle_t *eh );


/// Get the distortion coefficients
/**
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns New dynamic vector of distortion coefficients.
 */
VITAL_C_EXPORT
vital_eigen_matrixXx1d_t*
vital_camera_intrinsics_get_dist_coeffs( vital_camera_intrinsics_t *ci,
                                         vital_error_handle_t *eh );


/// Access the intrinsics as an upper triangular matrix
/**
 * \note This matrix includes the focal length, principal point,
 * aspect ratio, and skew, but does not model distortion
 *
 * \param ci Camera intrinsics object opaque pointer
 * \param eh Vital error handle structure pointer
 * \returns New 3x3 matrix of the intrinsics.
 */
VITAL_C_EXPORT
vital_eigen_matrix3x3d_t*
vital_camera_intrinsics_as_matrix( vital_camera_intrinsics_t *ci,
                                   vital_error_handle_t *eh );


/// Map normalized image coordinates into actual image coordinates
/**
 * This function applies both distortion and application of the
 * calibration matrix to map into actual image coordinates
 *
 * \param p Point to transform
 * \param eh Vital error handle structure pointer
 * \return new vector containing the transformed coordinate.
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_map_2d( vital_camera_intrinsics_t *ci,
                                vital_eigen_matrix2x1d_t *p,
                                vital_error_handle_t *eh );


/// Map a 3D point in camera coordinates into actual image coordinates
/**
 * \param p Point to transform
 * \param eh Vital error handle structure pointer
 * \return new vector containing the transformed coordinate.
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_map_3d( vital_camera_intrinsics_t *ci,
                                vital_eigen_matrix3x1d_t *p,
                                vital_error_handle_t *eh );


/// Unmap actual image coordinates back into normalized image coordinates
/**
 * \param p Point to transform
 * \param eh Vital error handle structure pointer
 * \returns New vector containing the untransformed coordinate.
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_unmap_2d( vital_camera_intrinsics_t *ci,
                                  vital_eigen_matrix2x1d_t *p,
                                  vital_error_handle_t *eh );


/// Map normalized image coordinates into distorted coordinates
/**
 * \param p Point to transform
 * \param eh Vital error handle structure pointer
 * \returns new vector containing
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_distort_2d( vital_camera_intrinsics_t *ci,
                                    vital_eigen_matrix2x1d_t *p,
                                    vital_error_handle_t *eh );


/// Unmap distorted normalized coordinates into normalized coordinates
/**
 * \param p Point to transform
 * \param eh Vital error handle structure pointer
 * \returns new 2D vector of the un-distorted coordinate.
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_undistort_2d( vital_camera_intrinsics_t *ci,
                                      vital_eigen_matrix2x1d_t *p,
                                      vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_CAMERA_INTRINSICS_H_
