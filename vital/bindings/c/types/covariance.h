// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
