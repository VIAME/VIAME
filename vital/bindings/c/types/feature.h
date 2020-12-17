// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::feature
 */

#ifndef VITAL_C_FEATURE_H_
#define VITAL_C_FEATURE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/covariance.h>
#include <vital/bindings/c/types/color.h>
#include <vital/bindings/c/types/eigen.h>

////////////////////////////////////////////////////////////////////////////////
// Generic functions + accessors

/// Base opaque structure for vital::feature
typedef struct vital_feature_s vital_feature_t;

/// Destroy a feature instance
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_feature_destroy( vital_feature_t *f, vital_error_handle_t *eh );

/// Get 2D image location
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns new eigen matrix instance containing the location
 */
VITAL_C_EXPORT
vital_eigen_matrix2x1d_t*
vital_feature_loc( vital_feature_t *f, vital_error_handle_t *eh );

/// Get feature magnitude
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns feature magnitude as a double
 */
VITAL_C_EXPORT
double
vital_feature_magnitude( vital_feature_t *f, vital_error_handle_t *eh );

/// Get feature scale
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns feature scale as a double
 */
VITAL_C_EXPORT
double
vital_feature_scale( vital_feature_t *f, vital_error_handle_t *eh );

/// Get feature angle
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns Feature angle as a double
 */
VITAL_C_EXPORT
double
vital_feature_angle( vital_feature_t *f, vital_error_handle_t *eh );

/// Get feature 2D covariance
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns New feature 2D covariance instance
 */
VITAL_C_EXPORT
vital_covariance_2d_t*
vital_feature_covar( vital_feature_t *f, vital_error_handle_t *eh );

/// Get Feature location's pixel color
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \returns new pixel color instance
 */
VITAL_C_EXPORT
vital_rgb_color_t*
vital_feature_color( vital_feature_t *f, vital_error_handle_t *eh );

/// Get the name of the instance's data type
/**
 * \param f Feature instance
 * \param eh Vital error handle instance
 * \return String name of the instance's data type
 */
VITAL_C_EXPORT
char const*
vital_feature_type_name( vital_feature_t const *f, vital_error_handle_t *eh );

////////////////////////////////////////////////////////////////////////////////
// Type specific constructors + functions

/// Declare type-specific feature functions
/**
 * \param T data type
 * \param S standard data type character symbol
 */
#define DECLARE_FEATURE_OPERATIONS( T, S ) \
\
/**
 * Create a new typed feature instance.
 *
 * \param loc Location of feature (copied)
 * \param mat Magnitude of feature
 * \param scale Scale of feature
 * \param angle Angle of feature
 * \param color Color of pixel at feature location
 * \param eh Vital error handle instance
 * \returns New feature instance
 */ \
VITAL_C_EXPORT \
vital_feature_t* \
vital_feature_##S##_new( vital_eigen_matrix2x1##S##_t *loc, T mag, T scale, \
                         T angle, vital_rgb_color_t *color, \
                         vital_error_handle_t *eh ); \
\
/**
 * Create a new typed feature instance with default parameters
 *
 * \param eh Vital error handle instance
 * \returns New feature instance
 */ \
VITAL_C_EXPORT \
vital_feature_t* \
vital_faeture_##S##_new_default( vital_error_handle_t *eh ); \
\
/**
 * Set feature location vector
 *
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param v Location to set (copied)
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_loc( vital_feature_t *f, \
                             vital_eigen_matrix2x1##S##_t *l, \
                             vital_error_handle_t *eh ); \
\
/**
 * Set feature magnitude
 *
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param mag Magnitude to set
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_magnitude( vital_feature_t *f, \
                                   T mag, \
                                   vital_error_handle_t *eh ); \
\
/**
 * Set feature scale
 *
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param scale scale to set
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_scale( vital_feature_t *f, \
                               T scale, \
                               vital_error_handle_t *eh ); \
\
/**
 * Set feature angle
 *
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param angle angle to set
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_angle( vital_feature_t *f, \
                               T angle, \
                               vital_error_handle_t *eh ); \
\
/**
 * Set feature covariance matrix
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param covar Covariance instance to set (copied)
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_covar( vital_feature_t *f, \
                               vital_covariance_2##S##_t *covar, \
                               vital_error_handle_t *eh ); \
\
/**
 * Set feature color
 *
 * Error occurs if the given data type of the feature provided does not match
 * the typed function called (failed a dynamic cast).
 *
 * \param f Feature instance to set location vector
 * \param c Color to set (copied)
 * \param eh Vital error handle instance
 */ \
VITAL_C_EXPORT \
void \
vital_feature_##S##_set_color( vital_feature_t *f, \
                               vital_rgb_color_t *c, \
                               vital_error_handle_t *eh ); \

DECLARE_FEATURE_OPERATIONS( double, d )
DECLARE_FEATURE_OPERATIONS( float,  f )

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_FEATURE_H_
