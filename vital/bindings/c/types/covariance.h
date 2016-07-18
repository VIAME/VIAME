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
 * \brief C interface to vital::covariance_ template classes
 */

#ifndef VITAL_C_COVARIANCE_H_
#define VITAL_C_COVARIANCE_H_

#include "eigen.h"


#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>


/// Declare Vital Covariance type functions (templates class)
/**
 * \tparam N Size of the covariance matrix
 * \tparam T Covariance data type
 * \tparam S Character suffix for naming convection
 */
#define DECLARE_VITAL_COVARIANCE_FUNCTIONS( N, T, S )                          \
                                                                               \
typedef struct vital_covariance_##N##S##_s vital_covariance_##N##S##_t;        \
                                                                               \
/**
 * Create a new covariance instance matrix
 */                                                                            \
VITAL_C_EXPORT                                                                 \
vital_covariance_##N##S##_t*                                                   \
vital_covariance_##N##S##_new( vital_error_handle_t *eh );                     \
                                                                               \
/**
 * Create a new covariance matrix - initialize to identity matrix times a scalar
 */                                                                            \
VITAL_C_EXPORT                                                                 \
vital_covariance_##N##S##_t*                                                   \
vital_covariance_##N##S##_new_from_scalar( T value,                            \
                                           vital_error_handle_t *eh );         \
                                                                               \
/**
 * Create new covariance matrix - from existing matrix of appropriate size
 *
 * An error occurs of the given matrix is not of the correct shape, yielding a
 * null pointer and a populated error handle.
 */                                                                               \
VITAL_C_EXPORT                                                                    \
vital_covariance_##N##S##_t*                                                      \
vital_covariance_##N##S##_new_from_matrix( vital_eigen_matrix##N##x##N##S##_t *m, \
                                           vital_error_handle_t *eh );            \
                                                                                  \
/**
 * Destroy covariance matrix instance
 */                                                                            \
VITAL_C_EXPORT                                                                 \
void                                                                           \
vital_covariance_##N##S##_destroy( vital_covariance_##N##S##_t *cov,           \
                                   vital_error_handle_t *eh );                 \
                                                                               \
/**
 * Create a new covariance matrix - from another covariance instance
 */                                                                            \
VITAL_C_EXPORT                                                                 \
vital_covariance_##N##S##_t*                                                   \
vital_covariance_##N##S##_new_copy( vital_covariance_##N##S##_t *other,        \
                                    vital_error_handle_t *eh );                \
                                                                               \
/**
 * Set the covariance data to that of another covariance instance (copy)
 */                                                                            \
VITAL_C_EXPORT                                                                 \
void                                                                           \
vital_covariance_##N##S##_set_covariance( vital_covariance_##N##S##_t *cov,    \
                                          vital_covariance_##N##S##_t *other,  \
                                          vital_error_handle_t *eh );          \
                                                                               \
/**
 * Convert this covariance into a new Eigen matrix instance.
 */                                                                            \
VITAL_C_EXPORT                                                                 \
vital_eigen_matrix##N##x##N##S##_t*                                            \
vital_covariance_##N##S##_to_matrix( vital_covariance_##N##S##_t *cov,         \
                                     vital_error_handle_t *eh );               \
                                                                               \
/**
 * Get the i-th row, j-th column
 */                                                                            \
VITAL_C_EXPORT                                                                 \
T                                                                              \
vital_covariance_##N##S##_get( vital_covariance_##N##S##_t *cov,               \
                               unsigned int i, unsigned int j,                 \
                               vital_error_handle_t *eh );                     \
                                                                               \
/**
 * Set the i-th row, j-th column
 */                                                                            \
VITAL_C_EXPORT                                                                 \
void                                                                           \
vital_covariance_##N##S##_set( vital_covariance_##N##S##_t *cov,               \
                               unsigned int i, unsigned int j, T value,        \
                               vital_error_handle_t *eh );


// Declare functions for sizes and types
DECLARE_VITAL_COVARIANCE_FUNCTIONS( 2, double, d )
DECLARE_VITAL_COVARIANCE_FUNCTIONS( 2, float,  f )
DECLARE_VITAL_COVARIANCE_FUNCTIONS( 3, double, d )
DECLARE_VITAL_COVARIANCE_FUNCTIONS( 3, float,  f )


#undef DECLARE_VITAL_COVARIANCE_FUNCTIONS


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_COVARIANCE_H_
