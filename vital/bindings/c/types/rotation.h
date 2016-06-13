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
 * \brief C Interface to \p vital::rotation_<T> class
 */

#ifndef VITAL_C_ROTATION_H_
#define VITAL_C_ROTATION_H_

#include "eigen.h"


#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>


/// Declare C rotation function for a given data type and type character suffix
/**
 * \param T The data storage type like double or float
 * \param S The character suffix to use for naming of functions.
 */
#define DECLARE_FUNCTIONS( T, S ) \
\
typedef struct vital_rotation_##S##_s vital_rotation_##S##_t; \
\
/**
 * Destroy rotation instance
 *
 * \param[in] rot The rotation instance to destroy
 * \param[in,out] eh Vital Error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_rotation_##S##_destroy( vital_rotation_##S##_t *rot, \
                              vital_error_handle_t *eh ); \
\
/**
 * Create new default rotation
 *
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_default( vital_error_handle_t *eh ); \
\
/**
 * Create new rotation from a 4D vector
 *
 * \param[in] q 4x1 matrix (vector) that is the quaternion to initialize from.
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_quaternion( vital_eigen_matrix4x1##S##_t *q, \
                                          vital_error_handle_t *eh ); \
\
/**
 * Create rotation for Rodrigues vector
 *
 * \param r 3x1 matrix (vector) that is the Rodrigues vector to initialize from.
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_rodrigues( vital_eigen_matrix3x1##S##_t *r, \
                                         vital_error_handle_t *eh ); \
\
/**
 * Create rotation from angle and axis
 *
 * \param angle Initial angle value.
 * \param axis Initial 3x1 axis vector
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_axis_angle( T angle, \
                                          vital_eigen_matrix3x1##S##_t *axis, \
                                          vital_error_handle_t *eh ); \
\
/**
 * Create rotation from yaw, pitch and roll
 *
 * \param yaw Initial yaw value
 * \param pitch Initial pitch value
 * \param roll Initial roll value
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_ypr( T yaw, T pitch, T roll, \
                                   vital_error_handle_t *eh ); \
\
/**
 * Create rotation from a 3x3 orthonormal matrix
 *
 * Requires an orthonormal matrix with +1 determinant.
 *
 * \param m Orthonormal matrix to initialize to.
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_matrix( vital_eigen_matrix3x3##S##_t *m, \
                                      vital_error_handle_t *eh ); \
\
/**
 * Convert a rotation into a new 3x3 matrix instance
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 * \return new Eigen::Matrix opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix3x3##S##_t* \
vital_rotation_##S##_to_matrix( vital_rotation_##S##_t *rot, \
                                vital_error_handle_t *eh ); \
\
/**
 * Get the axis of a rotation as a new matrix instance
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 * \return new Eigen::Matrix opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix3x1##S##_t* \
vital_rotation_##S##_axis( vital_rotation_##S##_t *rot, \
                           vital_error_handle_t *eh ); \
\
/**
 * Get the angle of the rotation in radians about the axis
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 * \return Angle value
 */ \
VITAL_C_EXPORT \
T \
vital_rotation_##S##_angle( vital_rotation_##S##_t *rot, \
                            vital_error_handle_t *eh ); \
\
/**
 * Get the quaternion vector from the rotation as a new 4x1 matrix
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 * \return new Eigen::Matrix opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix4x1##S##_t* \
vital_rotation_##S##_quaternion( vital_rotation_##S##_t *rot, \
                                 vital_error_handle_t *eh ); \
\
/**
 * Get the Rodrigues vector from the rotation as a new 3x1 matrix
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 * \return new Eigen::Matrix opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix3x1d_t* \
vital_rotation_##S##_rodrigues( vital_rotation_##S##_t *rot, \
                                vital_error_handle_t *eh ); \
\
/**
 * Get the rotation in its yaw, pitch and roll components.
 *
 * \param rot Pointer to rotation instance.
 * \param[out] yaw Pointer to set yaw value of the rotation.
 * \param[out] pitch Pointer to set pitch value of the rotation.
 * \param[out] roll Pointer to set roll value of the rotation.
 * \param[in,out] eh Vital Error handle instance.
 */ \
VITAL_C_EXPORT \
void \
vital_rotation_##S##_ypr( vital_rotation_##S##_t *rot, \
                          T *yaw, T *pitch, T *roll, \
                          vital_error_handle_t *eh ); \
\
/**
 * Get the inverse of a rotation as a new rotation instance
 *
 * \param rot Rotation instance pointer
 * \param[in,out] eh Vital Error handle instance
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_inverse( vital_rotation_##S##_t *rot, \
                              vital_error_handle_t *eh ); \
\
/**
 * Compose two rotations, returning a new rotation instance of the composition
 *
 * \param lhs Left-hand rotation instance for composition
 * \param rhs Right-hand rotation instance for composition
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_compose( vital_rotation_##S##_t *lhs, \
                              vital_rotation_##S##_t *rhs, \
                              vital_error_handle_t *eh ); \
\
/**
 * Rotate a vector using the given rotation, returning a new vector instance
 *
 * \param rot Rotation instance pointer
 * \param v 3x1 vector to rotate.
 * \param[in,out] eh Vital Error handle instance
 * \return new Eigen::Matrix opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix3x1##S##_t* \
vital_rotation_##S##_rotate_vector( vital_rotation_##S##_t *rot, \
                                    vital_eigen_matrix3x1##S##_t *v, \
                                    vital_error_handle_t *eh ); \
\
/**
 * Check rotation instance equality
 *
 * \param A Rotation to compare B to.
 * \param B Rotation to compare against A.
 * \param[in,out] eh Vital Error handle instance
 */ \
VITAL_C_EXPORT \
bool \
vital_rotation_##S##_are_equal( vital_rotation_##S##_t *A, \
                                vital_rotation_##S##_t *B, \
                                vital_error_handle_t *eh ); \
\
/**
 * Generate an interpolated rotation between A and B by a given fraction.
 *
 * \param A Rotation we are interpolating from
 * \param B rotation we are interpolating towards
 * \param f Fractional value describing the interpolation point between A and B.
 * \param[in,out] eh Vital Error handle instance
 * \return new vital::rotation_<T> opaque instance pointer
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t* \
vital_rotation_##S##_interpolate( vital_rotation_##S##_t *A, \
                                  vital_rotation_##S##_t *B, \
                                  T f, \
                                  vital_error_handle_t *eh ); \
\
/**
 * Generate N evenly interpolated rotations in between \c A and \c B
 *
 * \param[in] A Rotation we are interpolating from
 * \param[in] B rotation we are interpolating towards
 * \param[in] n Number of even interpolations in between A and B to generate
 * \param[in,out] eh Vital Error handle instance
 * \return Pointer to new rotation instances that are the interpolated results
 *         of size \c n.
 */ \
VITAL_C_EXPORT \
vital_rotation_##S##_t** \
vital_rotation_##S##_interpolated_rotations( vital_rotation_##S##_t *A, \
                                             vital_rotation_##S##_t *B, \
                                             size_t n, \
                                             vital_error_handle_t *eh );


DECLARE_FUNCTIONS( double, d )
DECLARE_FUNCTIONS( float,  f )


#undef DECLARE_FUNCTIONS


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ROTATION_H_
