// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface implementation to vital::covariance_ template classes
 */

#include "covariance.h"

#include <sstream>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/types/covariance.h>

/// Define Vital Covariance type functions (templates class)
/**
 * \tparam N Size of the covariance matrix
 * \tparam T Covariance data type
 * \tparam S Character suffix for naming convection
 */
#define DEFINE_VITAL_COVARIANCE_FUNCTIONS( N, T, S )                                 \
                                                                                     \
vital_covariance_##N##S##_t*                                                         \
vital_covariance_##N##S##_new( vital_error_handle_t *eh )                            \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".new", eh,                                            \
    return reinterpret_cast<vital_covariance_##N##S##_t*>( new cov_t() );            \
  );                                                                                 \
  return 0;                                                                          \
}                                                                                    \
                                                                                     \
/** Create a new covariance matrix - initialize to identity matrix times a scalar */ \
vital_covariance_##N##S##_t*                                                         \
vital_covariance_##N##S##_new_from_scalar( T value,                                  \
                                           vital_error_handle_t *eh )                \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance" #N #S ".new_from_scalar", eh,                                 \
    return reinterpret_cast<vital_covariance_##N##S##_t*>( new cov_t( value ) );     \
  );                                                                                 \
  return 0;                                                                          \
}                                                                                    \
                                                                                     \
/** Create new covariance matrix - from existing matrix of appropriate size */       \
vital_covariance_##N##S##_t*                                                         \
vital_covariance_##N##S##_new_from_matrix( vital_eigen_matrix##N##x##N##S##_t *m,    \
                                           vital_error_handle_t *eh )                \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  typedef Eigen::Matrix< T, N, N > matrix_t;                                         \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".new_from_matrix", eh,                                \
    /* Make sure matrix is the correct shape after cast. Could have been
     * given a mis-masted opaque pointer to instance (e.g. python ctypes
     * trickery)
     */                                                                              \
    REINTERP_TYPE( matrix_t, m, mat );                                               \
    if ( ! ( mat->rows() == N && mat->cols() == N ) ) {                              \
      std::stringstream ss;                                                          \
      ss << "Invalid input matrix shape (" << mat->rows() << ", " << mat->cols()     \
         << ")";                                                                     \
      throw ss.str().c_str();                                                        \
    }                                                                                \
    return reinterpret_cast<vital_covariance_##N##S##_t*>( new cov_t( *mat ) );      \
  );                                                                                 \
  return 0;                                                                          \
}                                                                                    \
                                                                                     \
/** Create a new covariance matrix - from another covariance instance */             \
vital_covariance_##N##S##_t*                                                         \
vital_covariance_##N##S##_new_copy( vital_covariance_##N##S##_t *other,              \
                                    vital_error_handle_t *eh )                       \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".new_copy", eh,                                       \
    REINTERP_TYPE(cov_t, other, cov_ptr );                                           \
    return reinterpret_cast<vital_covariance_##N##S##_t*>( new cov_t( *cov_ptr ) );  \
  );                                                                                 \
  return 0;                                                                          \
}                                                                                    \
                                                                                     \
/** Destroy covariance matrix instance */                                            \
void                                                                                 \
vital_covariance_##N##S##_destroy( vital_covariance_##N##S##_t *cov,                 \
                                   vital_error_handle_t *eh )                        \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".destroy", eh,                                        \
    REINTERP_TYPE( cov_t, cov, cov_ptr );                                            \
    delete cov_ptr;                                                                  \
  );                                                                                 \
}                                                                                    \
                                                                                     \
/** Set the covariance data to that of another covariance instance (copy) */         \
void                                                                                 \
vital_covariance_##N##S##_set_covariance( vital_covariance_##N##S##_t *cov,          \
                                          vital_covariance_##N##S##_t *other,        \
                                          vital_error_handle_t *eh )                 \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".set_covariance", eh,                                 \
    REINTERP_TYPE( cov_t, cov, cov_ptr );                                            \
    REINTERP_TYPE( cov_t, other, other_cov_ptr );                                    \
    (*cov_ptr) = (*other_cov_ptr);                                                   \
  );                                                                                 \
}                                                                                    \
                                                                                     \
/** Convert this covariance into a new Eigen matrix instance. */                     \
vital_eigen_matrix##N##x##N##S##_t*                                                  \
vital_covariance_##N##S##_to_matrix( vital_covariance_##N##S##_t *cov,               \
                                     vital_error_handle_t *eh )                      \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  typedef Eigen::Matrix<T, N, N> matrix_t;                                           \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".to_matrix", eh,                                      \
    REINTERP_TYPE( cov_t, cov, cov_ptr );                                            \
    matrix_t *m = new matrix_t( cov_ptr->matrix() );                                 \
    return reinterpret_cast<vital_eigen_matrix##N##x##N##S##_t*>( m );               \
  );                                                                                 \
  return 0;                                                                          \
}                                                                                    \
                                                                                     \
/** Get the i-th row, j-th column */                                                 \
T                                                                                    \
vital_covariance_##N##S##_get( vital_covariance_##N##S##_t *cov,                     \
                               unsigned int i, unsigned int j,                       \
                               vital_error_handle_t *eh )                            \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".get", eh,                                            \
    REINTERP_TYPE( cov_t, cov, cov_ptr );                                            \
    return (*cov_ptr)(i, j);                                                         \
  );                                                                                 \
  return 0.;                                                                         \
}                                                                                    \
                                                                                     \
/** Set the i-th row, j-th column */                                                 \
void                                                                                 \
vital_covariance_##N##S##_set( vital_covariance_##N##S##_t *cov,                     \
                               unsigned int i, unsigned int j, T value,              \
                               vital_error_handle_t *eh )                            \
{                                                                                    \
  typedef kwiver::vital::covariance_<N, T> cov_t;                                    \
  STANDARD_CATCH(                                                                    \
    "vital_covariance_" #N #S ".set", eh,                                            \
    REINTERP_TYPE( cov_t, cov, cov_ptr );                                            \
    (*cov_ptr)(i, j) = value;                                                        \
  );                                                                                 \
}

// Define functions for sizes and types
DEFINE_VITAL_COVARIANCE_FUNCTIONS( 2, double, d )
DEFINE_VITAL_COVARIANCE_FUNCTIONS( 2, float,  f )
DEFINE_VITAL_COVARIANCE_FUNCTIONS( 3, double, d )
DEFINE_VITAL_COVARIANCE_FUNCTIONS( 3, float,  f )

#undef DEFINE_VITAL_COVARIANCE_FUNCTIONS
