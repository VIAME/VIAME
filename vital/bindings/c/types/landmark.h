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
 * \brief Landmark C interface
 */

#ifndef VITAL_C_LANDMARK_H_
#define VITAL_C_LANDMARK_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/covariance.h>
#include <vital/bindings/c/types/eigen.h>
#include <vital/bindings/c/types/color.h>


/// VITAL Landmark opaque structure
typedef struct vital_landmark_s vital_landmark_t;


// Type independent functions //////////////////////////////////////////////////

/// Destroy landmark instance
/**
 * \param l Landmark instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_landmark_destroy( vital_landmark_t *l, vital_error_handle_t *eh );


/// Clone the given landmark, returning a new instance.
/**
 * \param l Landmark instance to clone
 * \param eh Vital error handle instance
 * \returns New landmark instance that is the clone of \c l.
 */
VITAL_C_EXPORT
vital_landmark_t*
vital_landmark_clone( vital_landmark_t *l, vital_error_handle_t *eh );


/// Get the name of the stored data type
/**
 * \param l Landmark instance
 * \param eh Vital error handle instance
 * \returns String type identifier
 */
VITAL_C_EXPORT
char const*
vital_landmark_type_name( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the world location of a landmark as a new 3x1 matrix
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \returns New 3x1 Eigen matrix instance
 */
VITAL_C_EXPORT
vital_eigen_matrix3x1d_t*
vital_landmark_loc( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the scale or a landmark
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \returns Scale value
 */
VITAL_C_EXPORT
double
vital_landmark_scale( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the normal vector for a landmark as a new 3x1 matrix
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \return New 3x1 Eigen matrix instance
 */
VITAL_C_EXPORT
vital_eigen_matrix3x1d_t*
vital_landmark_normal( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the covariance of a landmark
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \returns New 3D covariance instance
 */
VITAL_C_EXPORT
vital_covariance_3d_t*
vital_landmark_covariance( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the RGB color of a landmark
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \returns New color instance
 */
VITAL_C_EXPORT
vital_rgb_color_t*
vital_landmark_color( vital_landmark_t const *l, vital_error_handle_t *eh );


/// Get the observations of a landmark
/**
 * \param l landmark instance
 * \param eh Vital error handle instance
 * \returns Observations value
 */
VITAL_C_EXPORT
unsigned
vital_landmark_observations( vital_landmark_t const *l, vital_error_handle_t *eh );


// Type-dependent functions ////////////////////////////////////////////////////

/// Declare type-specific landmark functions
/**
 * \param T data type
 * \param S standard data type character symbol
 */
#define DECLARE_TYPED_OPERATIONS( T, S ) \
\
/**
 * Create a new instance of a landmark from a 3D coordinate
 *
 * Scale is set to 1 by default.
 *
 * \param loc 3D location of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
vital_landmark_t* \
vital_landmark_##S##_new( vital_eigen_matrix3x1##S##_t const *loc, \
                          vital_error_handle_t *eh ); \
\
/**
 * Create a new instance of a landmark from a coordinate with a scale
 *
 * \param loc 3D location of the landmark
 * \param scale Scale of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
vital_landmark_t* \
vital_landmark_##S##_new_scale( vital_eigen_matrix3x1##S##_t const *loc, \
                                T const scale, vital_error_handle_t *eh ); \
\
/**
 * Create a new default instance of a landmark
 *
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
vital_landmark_t* \
vital_landmark_##S##_new_default( vital_error_handle_t *eh ); \
\
/**
 * Set 3D location of a landmark instance
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param loc New 3D location for the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_loc( vital_landmark_t *l, \
                              vital_eigen_matrix3x1##S##_t const *loc, \
                              vital_error_handle_t *eh ); \
\
/**
 * Set the scale of the landmark
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param scale New scale of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_scale( vital_landmark_t *l, T scale, \
                                vital_error_handle_t *eh ); \
\
/**
 * Set the normal vector of the landmark
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param normal New 3D normal vector of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_normal( vital_landmark_t *l, \
                                 vital_eigen_matrix3x1##S##_t const *normal, \
                                 vital_error_handle_t *eh ); \
\
/**
 * Set the covariance of the landmark
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param covar New 3D covariance of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_covar( vital_landmark_t *l, \
                                vital_covariance_3##S##_t const *covar, \
                                vital_error_handle_t *eh ); \
\
/**
 * Set the color of the landmark
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param c New color of the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_color( vital_landmark_t *l, \
                                vital_rgb_color_t const *c, \
                                vital_error_handle_t *eh ); \
\
/**
 * Set the observations of the landmark
 *
 * This may error if the underlying landmark cannot be dynamic cast to the
 * requested type (error code 1).
 *
 * \param l landmark instance
 * \param observations New observations value for the landmark
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_landmark_##S##_set_observations( vital_landmark_t *l, \
                                       unsigned observations, \
                                       vital_error_handle_t *eh );


DECLARE_TYPED_OPERATIONS( double, d )
DECLARE_TYPED_OPERATIONS( float,  f )

#undef DECLARE_TYPED_OPERATIONS


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_LANDMARK_H_
