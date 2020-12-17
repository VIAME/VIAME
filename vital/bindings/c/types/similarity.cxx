// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#include "similarity.h"

#include <sstream>

#include <vital/types/rotation.h>
#include <vital/types/similarity.h>

#include <vital/bindings/c/helpers/c_utils.h>

using namespace kwiver;

#define DEFINE_TYPED_OPERATIONS( T, S ) \
\
/* Create a new similarity instance */ \
vital_similarity_##S##_t* \
vital_similarity_##S##_new( T s, vital_rotation_##S##_t const *r, \
                            vital_eigen_matrix3x1##S##_t const *t, \
                            vital_error_handle_t *eh ) \
{ \
  typedef vital::rotation_< T > rotation_t; \
  typedef Eigen::Matrix< T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( rotation_t const, r, r_ptr ); \
    REINTERP_TYPE( matrix_t const, t, t_ptr ); \
    return reinterpret_cast< vital_similarity_##S##_t* >( \
      new vital::similarity_< T >( s, *r_ptr, *t_ptr ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Create a new similarity instance with default initial value */ \
vital_similarity_##S##_t* \
vital_similarity_##S##_new_default( vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    return reinterpret_cast< vital_similarity_##S##_t* >( \
      new vital::similarity_< T >() \
    ); \
  ); \
  return NULL; \
} \
\
/* Destroy a similarity instance */ \
void \
vital_similarity_##S##_destroy( vital_similarity_##S##_t *s, \
                                vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T>, s, s_ptr ); \
    delete s_ptr; \
  ); \
} \
\
/* Get the scale factor of a similarity instance */ \
T \
vital_similarity_##S##_scale( vital_similarity_##S##_t const *sim, \
                              vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T> const, sim, s_ptr ); \
    return s_ptr->scale(); \
  ); \
  return 0; \
} \
\
/* Get the rotation of a similarity instance */ \
vital_rotation_##S##_t* \
vital_similarity_##S##_rotation( vital_similarity_##S##_t const *sim, \
                                 vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T> const, sim, s_ptr ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( \
      new vital::rotation_<T>( s_ptr->rotation() ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Get the translation of a similarity instance */ \
vital_eigen_matrix3x1##S##_t* \
vital_similarity_##S##_translation( vital_similarity_##S##_t const *sim, \
                                    vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix< T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T> const, sim, s_ptr ); \
    return reinterpret_cast< vital_eigen_matrix3x1##S##_t* >( \
      new matrix_t( s_ptr->translation() ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Compute the inverse of a similarity, returning a new similarity instance */ \
vital_similarity_##S##_t* \
vital_similarity_##S##_inverse( vital_similarity_##S##_t const *sim, \
                                vital_error_handle_t *eh ) \
{ \
  typedef vital::similarity_<T> sim_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( sim_t const, sim, s_ptr ); \
    return reinterpret_cast< vital_similarity_##S##_t* >( \
      new sim_t( s_ptr->inverse() ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Compose two similarities */ \
vital_similarity_##S##_t* \
vital_similarity_##S##_compose( vital_similarity_##S##_t const *s_lhs, \
                                vital_similarity_##S##_t const *s_rhs, \
                                vital_error_handle_t *eh ) \
{ \
  typedef vital::similarity_<T> sim_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( sim_t const, s_lhs, s_lhs_ptr ); \
    REINTERP_TYPE( sim_t const, s_rhs, s_rhs_ptr ); \
    sim_t *r_ptr = new sim_t( (*s_lhs_ptr) * (*s_rhs_ptr) ); \
    return reinterpret_cast< vital_similarity_##S##_t* >( r_ptr ); \
  ); \
  return NULL; \
} \
 \
/* Transform a vector */ \
vital_eigen_matrix3x1##S##_t* \
vital_similarity_##S##_vector_transform( vital_similarity_##S##_t const *s, \
                                         vital_eigen_matrix3x1##S##_t const *rhs, \
                                         vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix< T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T> const, s, s_ptr ); \
    REINTERP_TYPE( matrix_t const, rhs, rhs_ptr ); \
    return reinterpret_cast< vital_eigen_matrix3x1##S##_t* >( \
      new matrix_t( (*s_ptr) * (*rhs_ptr) ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Test equality between two similarities */ \
bool \
vital_similarity_##S##_are_equal( vital_similarity_##S##_t const *s_lhs, \
                                  vital_similarity_##S##_t const *s_rhs, \
                                  vital_error_handle_t *eh ) \
{ \
  typedef vital::similarity_<T> sim_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( sim_t const, s_lhs, lhs_ptr ); \
    REINTERP_TYPE( sim_t const, s_rhs, rhs_ptr ); \
    return (*lhs_ptr) == (*rhs_ptr); \
  ); \
  return false; \
} \
\
/* Convert a similarity into a 4x4 matrix */ \
vital_eigen_matrix4x4##S##_t* \
vital_similarity_##S##_to_matrix4x4( vital_similarity_##S##_t const *s, \
                                     vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix< T, 4, 4 > matrix_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( vital::similarity_<T> const, s, s_ptr ); \
    return reinterpret_cast< vital_eigen_matrix4x4##S##_t* >( \
      new matrix_t( s_ptr->matrix() ) \
    ); \
  ); \
  return NULL; \
} \
\
/* Create a similarity from a 4x4 matrix */ \
vital_similarity_##S##_t* \
vital_similarity_##S##_from_matrix4x4( vital_eigen_matrix4x4##S##_t const *m, \
                                       vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix< T, 4, 4 > matrix_t; \
  STANDARD_CATCH( \
    "vital_similarity_" #S "", eh, \
    REINTERP_TYPE( matrix_t const, m, m_ptr ); \
    return reinterpret_cast< vital_similarity_##S##_t* >( \
      new vital::similarity_<T>( *m_ptr ) \
    ); \
  ); \
  return NULL; \
}

DEFINE_TYPED_OPERATIONS( double, d )
DEFINE_TYPED_OPERATIONS( float,  f )

#undef DEFINE_TYPED_OPERATIONS
