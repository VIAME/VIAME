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
 * \brief Implementation of C Interface to \p vital::rotation_<T> class
 */

#include "rotation.h"

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/types/rotation.h>


/// Define C rotation functions for a given data type and type character suffix
/**
 * \param T The data storage type like double or float
 * \param S The character suffix to use for naming of functions.
 */
#define DEFINE_FUNCTIONS( T, S )\
\
/* Destroy rotation instance */ \
void \
vital_rotation_##S##_destroy( vital_rotation_##S##_t *rot, \
                              vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_< T > rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".destroy", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    delete rot_ptr; \
  ); \
} \
\
/* Create new default rotation */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_default( vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_default", eh, \
    return reinterpret_cast< vital_rotation_##S##_t* >( new rotation_t() ); \
  ); \
  return 0; \
} \
\
/* Create new rotation from a 4D vector */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_quaternion( vital_eigen_matrix4x1##S##_t *q, \
                                          vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 4, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_from_quaternion", eh, \
    REINTERP_TYPE( matrix_t, q, mat_ptr ); \
    return reinterpret_cast<vital_rotation_##S##_t*>( \
      new rotation_t( *mat_ptr ) \
    ); \
  ); \
  return 0; \
} \
\
/* Create rotation for Rodrigues vector */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_rodrigues( vital_eigen_matrix3x1##S##_t *r, \
                                         vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_from_rodrigues", eh, \
    REINTERP_TYPE( matrix_t, r, r_ptr ); \
    return reinterpret_cast<vital_rotation_##S##_t*>( \
      new rotation_t( *r_ptr ) \
    ); \
  ); \
  return 0; \
} \
\
/* Create rotation from angle and axis */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_axis_angle( T angle, \
                                          vital_eigen_matrix3x1##S##_t *axis, \
                                          vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_from_axis_angle", eh, \
    REINTERP_TYPE( matrix_t, axis, axis_ptr ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( \
      new rotation_t( angle, *axis_ptr ) \
    ); \
  ); \
  return 0; \
} \
\
/* Create rotation from yaw, pitch and roll */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_ypr( T yaw, T pitch, T roll, \
                                   vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_from_ypr", eh, \
    return reinterpret_cast< vital_rotation_##S##_t* >( \
      new rotation_t( yaw, pitch, roll ) \
    );\
  ); \
  return 0; \
} \
\
/* Create rotation from a 3x3 orthonormal matrix */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_new_from_matrix( vital_eigen_matrix3x3##S##_t *m, \
                                      vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 3 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".new_from_matrix", eh, \
    REINTERP_TYPE( matrix_t, m, m_ptr ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( \
      new rotation_t( *m_ptr ) \
    ); \
  ); \
  return 0; \
} \
\
/* Convert a rotation into a new 3x3 matrix instance */ \
vital_eigen_matrix3x3##S##_t* \
vital_rotation_##S##_to_matrix( vital_rotation_##S##_t *rot, \
                                vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 3 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".to_matrix", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    matrix_t mat = (*rot_ptr); \
    return reinterpret_cast< vital_eigen_matrix3x3##S##_t* >( \
      new matrix_t( mat ) \
    ); \
  ); \
  return 0; \
} \
\
/* Get the axis of a rotation as a new matrix instance */ \
vital_eigen_matrix3x1##S##_t* \
vital_rotation_##S##_axis( vital_rotation_##S##_t *rot, \
                           vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".axis", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    return reinterpret_cast< vital_eigen_matrix3x1##S##_t* >( \
      new matrix_t( rot_ptr->axis() ) \
    ); \
  ); \
  return 0; \
} \
\
/* Get the angle of the rotation in radians about the axis */ \
T \
vital_rotation_##S##_angle( vital_rotation_##S##_t *rot, \
                            vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".angle", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    return rot_ptr->angle(); \
  ); \
  return 0; \
} \
\
/* Get the quaternion vector from the rotation as a new 4x1 matrix */ \
vital_eigen_matrix4x1##S##_t* \
vital_rotation_##S##_quaternion( vital_rotation_##S##_t *rot, \
                                 vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 4, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".quaternion", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    matrix_t *mat_ptr = new matrix_t( rot_ptr->quaternion().coeffs() ); \
    return reinterpret_cast< vital_eigen_matrix4x1##S##_t* >( mat_ptr ); \
  ); \
  return 0; \
} \
\
/* Get the Rodrigues vector from the rotation as a new 3x1 matrix */ \
vital_eigen_matrix3x1d_t* \
vital_rotation_##S##_rodrigues( vital_rotation_##S##_t *rot, \
                                vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".rodrigues", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    matrix_t *mat_ptr = new matrix_t( rot_ptr->rodrigues() ); \
    return reinterpret_cast< vital_eigen_matrix3x1d_t* >( mat_ptr ); \
  ); \
  return 0; \
} \
\
/* Get the rotation in its yaw, pitch and roll components. */ \
void \
vital_rotation_##S##_ypr( vital_rotation_##S##_t *rot, \
                          T *yaw, T *pitch, T *roll, \
                          vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".ypr", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    rot_ptr->get_yaw_pitch_roll( *yaw, *pitch, *roll ); \
  ); \
} \
\
/* Get the inverse of a rotation as a new rotation instance */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_inverse( vital_rotation_##S##_t *rot, \
                              vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".inverse", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( \
      new rotation_t( rot_ptr->inverse() ) \
    ); \
  ); \
  return 0; \
} \
\
/* Compose two rotations, returning a new rotation instance of the composition */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_compose( vital_rotation_##S##_t *lhs, \
                              vital_rotation_##S##_t *rhs, \
                              vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".compose", eh, \
    REINTERP_TYPE( rotation_t, lhs, lhs_ptr ); \
    REINTERP_TYPE( rotation_t, rhs, rhs_ptr ); \
    rotation_t *prod_ptr = new rotation_t( (*lhs_ptr) * (*rhs_ptr) ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( prod_ptr ); \
  ); \
  return 0; \
} \
\
/* Rotate a vector using the given rotation, returning a new vector instance */ \
vital_eigen_matrix3x1##S##_t* \
vital_rotation_##S##_rotate_vector( vital_rotation_##S##_t *rot, \
                                    vital_eigen_matrix3x1##S##_t *v, \
                                    vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  typedef Eigen::Matrix< T, 3, 1 > matrix_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".rotate_vector", eh, \
    REINTERP_TYPE( rotation_t, rot, rot_ptr ); \
    REINTERP_TYPE( matrix_t, v, v_ptr ); \
    matrix_t *prod_ptr = new matrix_t( (*rot_ptr) * (*v_ptr) ); \
    return reinterpret_cast< vital_eigen_matrix3x1##S##_t* >( prod_ptr ); \
  ); \
  return 0; \
} \
\
/* Check rotation instance equality */ \
bool \
vital_rotation_##S##_are_equal( vital_rotation_##S##_t *A, \
                                vital_rotation_##S##_t *B, \
                                vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".are_equal", eh, \
    REINTERP_TYPE( rotation_t, A, A_ptr ); \
    REINTERP_TYPE( rotation_t, B, B_ptr ); \
    return ( (*A_ptr) == (*B_ptr) ); \
  ); \
  return false; \
} \
\
/* Generate an interpolated rotation between A and B by a given fraction */ \
vital_rotation_##S##_t* \
vital_rotation_##S##_interpolate( vital_rotation_##S##_t *A, \
                                  vital_rotation_##S##_t *B, \
                                  T f, \
                                  vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".interpolate", eh, \
    REINTERP_TYPE( rotation_t, A, A_ptr ); \
    REINTERP_TYPE( rotation_t, B, B_ptr ); \
    rotation_t *i_ptr = new rotation_t( \
      kwiver::vital::interpolate_rotation( *A_ptr, *B_ptr, f ) \
    ); \
    return reinterpret_cast< vital_rotation_##S##_t* >( i_ptr ); \
  ); \
  return 0; \
} \
\
/* Generate N evenly interpolated rotations in between \c A and \c B */ \
vital_rotation_##S##_t** \
vital_rotation_##S##_interpolated_rotations( vital_rotation_##S##_t *A, \
                                             vital_rotation_##S##_t *B, \
                                             size_t n, \
                                             vital_error_handle_t *eh ) \
{ \
  typedef kwiver::vital::rotation_<T> rotation_t; \
  STANDARD_CATCH( \
    "vital_rotation_" #S ".interpolateD_rotations", eh, \
    REINTERP_TYPE( rotation_t, A, A_ptr ); \
    REINTERP_TYPE( rotation_t, B, B_ptr ); \
    /* Generate vector of interpolated rotations */ \
    std::vector< rotation_t > i_rotations; \
    kwiver::vital::interpolated_rotations( *A_ptr, *B_ptr, n, i_rotations ); \
    /* Convert into pointer array */ \
    vital_rotation_##S##_t **rots = \
      (vital_rotation_##S##_t**)malloc(sizeof(vital_rotation_##S##_t*) * n); \
    for (size_t i = 0; i < i_rotations.size(); ++i ) \
    { \
      rots[i] = reinterpret_cast< vital_rotation_##S##_t* >( \
        new rotation_t( i_rotations[i] ) \
      ); \
    } \
    return rots; \
  ); \
  return 0; \
}


DEFINE_FUNCTIONS( double, d )
DEFINE_FUNCTIONS( float,  f )


#undef DEFINE_FUNCTIONS
