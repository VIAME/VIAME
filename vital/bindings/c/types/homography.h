// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#ifndef VITAL_C_HOMOGRAPHY_H_
#define VITAL_C_HOMOGRAPHY_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/eigen.h>
#include <vital/bindings/c/vital_c_export.h>

/// Opaque type structure
typedef struct vital_homography_s vital_homography_t;

/// Destroy a homography instance
/**
 * \param h Homography instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_homography_destroy( vital_homography_t *h, vital_error_handle_t *eh );

/// Get the name of the descriptor instance's underlying data type
/**
 * \param h Homography instance to destroy
 * \param eh Vital error handle instance
 * \returns Data type string name
 */
VITAL_C_EXPORT
char const*
vital_homography_type_name( vital_homography_t const *h,
                            vital_error_handle_t *eh );

/// Create a clone of a homography instance
/**
 * \param h Homography instance to destroy
 * \param eh Vital error handle instance
 * \returns New homography instance that is the clone of \c h.
 */
VITAL_C_EXPORT
vital_homography_t*
vital_homography_clone( vital_homography_t const *h, vital_error_handle_t *eh );

/// Get a new homography instance that is normalized
/**
 * \param h Homography instance to destroy
 * \param eh Vital error handle instance
 * \returns New homography instance that is the normalized version of \c h.
 */
VITAL_C_EXPORT
vital_homography_t*
vital_homography_normalize( vital_homography_t const *h,
                            vital_error_handle_t *eh );

/// Get a new homography instance that has been inverted
/**
 * \param h Homography instance to destroy
 * \param eh Vital error handle instance
 * \return New homography instance that is the inverse of \c h.
 */
VITAL_C_EXPORT
vital_homography_t*
vital_homography_inverse( vital_homography_t const *h,
                          vital_error_handle_t *eh );

////////////////////////////////////////////////////////////////////////////////
// Type specific functions and constructors

#define DECLARE_TYPED_OPERATIONS( T, S ) \
\
/** New identity homography
 * \param eh Vital error handle instance
 * \returns New homography instance with the identity transformation.
 */ \
VITAL_C_EXPORT \
vital_homography_t* \
vital_homography_##S##_new_identity( vital_error_handle_t *eh ); \
\
/** New homography from a provided transformation matrix
 * \param m Transformation matrix to initialize with
 * \param eh Vital error handle instance
 * \returns New homography instance with the transformation matrix \c m.
 */ \
VITAL_C_EXPORT \
vital_homography_t* \
vital_homography_##S##_new_from_matrix( vital_eigen_matrix3x3##S##_t const *m, \
                                        vital_error_handle_t *eh ); \
\
/** Get a homography's transformation as a new matrix
 * \param h Homography instance
 * \param eh Vital error handle instance
 * \return A copy of the transformation matrix of \c h.
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix3x3##S##_t* \
vital_homography_##S##_as_matrix( vital_homography_t const *h, \
                                  vital_error_handle_t *eh ); \
\
/** Map a 2D point using this homography, returning a new matrix instance
 *
 * The result point to infinity, resulting in an error code of 1.
 *
 * \param h Homography instance
 * \param p 2D point to map using the homography transformation \c h.
 * \param eh Vital error handle instance
 * \returns New point instance that is the transformation of \c p through \c h.
 */ \
VITAL_C_EXPORT \
vital_eigen_matrix2x1##S##_t* \
vital_homography_##S##_map_point( vital_homography_t const *h, \
                                  vital_eigen_matrix2x1##S##_t const *p, \
                                  vital_error_handle_t *eh ); \
\
/** Multiply one homography against another, returning a new product homography
 *
 * This effectively multiplies the underlying matrices together.
 *
 * \param lhs Left-hand side value of the multiplication operation
 * \param rhs Right-hand side value of the multiplication operation
 * \param eh Vital error handle instance
 * \returns New homography instance that is the result of multiplying the
 *   transformation matrices of \c lhs and \c rhs.
 */ \
VITAL_C_EXPORT \
vital_homography_t* \
vital_homography_##S##_multiply( vital_homography_t const *lhs, \
                                 vital_homography_t const *rhs, \
                                 vital_error_handle_t *eh );

DECLARE_TYPED_OPERATIONS( double, d )
DECLARE_TYPED_OPERATIONS( float,  f )

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_HOMOGRAPHY_H_
