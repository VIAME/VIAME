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
 * \brief File description here.
 */

#include "landmark.h"

#include <vital/bindings/c/helpers/landmark.h>


namespace kwiver {
namespace vital_c {

/// Cache for landmark shared pointers that exit the C++ barrier
SharedPointerCache< vital::landmark, vital_landmark_t >
  LANDMARK_SPTR_CACHE( "landmark" );

}
}


using namespace kwiver;


// Type independent functions //////////////////////////////////////////////////

/// Destroy landmark instance
void
vital_landmark_destroy( vital_landmark_t *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_destroy", eh,
    vital_c::LANDMARK_SPTR_CACHE.erase( l );
  );
}


/// Clone the given landmark, returning a new instance.
vital_landmark_t*
vital_landmark_clone( vital_landmark_t *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_clone", eh,

    auto l_clone = vital_c::LANDMARK_SPTR_CACHE.get( l )->clone();
    vital_c::LANDMARK_SPTR_CACHE.store( l_clone );
    return reinterpret_cast< vital_landmark_t* >( l_clone.get() );

  );
  return NULL;
}


/// Get the name of the stored data type
char const*
vital_landmark_type_name( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_type_name", eh,
    return vital_c::LANDMARK_SPTR_CACHE.get( l )->data_type().name();
  );
  return 0;
}


/// Get the world location of a landmark
vital_eigen_matrix3x1d_t*
vital_landmark_loc( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  typedef Eigen::Matrix<double, 3, 1> matrix_t;
  STANDARD_CATCH(
    "vital_landmark_loc", eh,
    return reinterpret_cast< vital_eigen_matrix3x1d_t* >(
      new matrix_t( vital_c::LANDMARK_SPTR_CACHE.get( l )->loc() )
    );
  );
  return 0;
}


/// Get the scale or a landmark
double
vital_landmark_scale( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_scale", eh,
    return vital_c::LANDMARK_SPTR_CACHE.get( l )->scale();
  );
  return 0;
}


/// Get the normal vector for a landmark
vital_eigen_matrix3x1d_t*
vital_landmark_normal( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  typedef Eigen::Matrix<double, 3, 1> matrix_t;
  STANDARD_CATCH(
    "vital_landmark_normal", eh,
    return reinterpret_cast< vital_eigen_matrix3x1d_t* >(
      new matrix_t( vital_c::LANDMARK_SPTR_CACHE.get( l )->normal() )
    );
  );
  return 0;
}


/// Get the covariance of a landmark
vital_covariance_3d_t*
vital_landmark_covariance( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_covariance", eh,
    return reinterpret_cast< vital_covariance_3d_t* >(
      new vital::covariance_3d( vital_c::LANDMARK_SPTR_CACHE.get( l )->covar() )
    );
  );
  return 0;
}


/// Get the RGB color of a landmark
vital_rgb_color_t*
vital_landmark_color( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_color", eh,
    return reinterpret_cast< vital_rgb_color_t* >(
      new vital::rgb_color( vital_c::LANDMARK_SPTR_CACHE.get( l )->color() )
    );
  );
  return 0;
}


/// Get the observations of a landmark
unsigned
vital_landmark_observations( vital_landmark_t const *l, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_observations", eh,
    return vital_c::LANDMARK_SPTR_CACHE.get( l )->observations();
  );
  return 0;
}


// Type dependent functions ////////////////////////////////////////////////////

#define DEFINE_TYPED_OPERATIONS( T, S ) \
\
/** Create a new instance of a landmark from a 3D coordinate */ \
vital_landmark_t* \
vital_landmark_##S##_new( vital_eigen_matrix3x1##S##_t const *loc, \
                          vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix<T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_new", eh, \
    REINTERP_TYPE( matrix_t const, loc, loc_ptr ); \
    auto l_sptr = std::make_shared< vital::landmark_<T> >( *loc_ptr ); \
    vital_c::LANDMARK_SPTR_CACHE.store( l_sptr ); \
    return reinterpret_cast< vital_landmark_t* >( l_sptr.get() ); \
  ); \
  return 0; \
} \
\
/** Create a new instance of a landmark from a coordinate with a scale */ \
vital_landmark_t* \
vital_landmark_##S##_new_scale( vital_eigen_matrix3x1##S##_t const *loc, \
                                T const scale, vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix<T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_new_scaled", eh, \
    REINTERP_TYPE( matrix_t const, loc, loc_ptr ); \
    auto l_sptr = std::make_shared< vital::landmark_<T> >( *loc_ptr, scale ); \
    vital_c::LANDMARK_SPTR_CACHE.store( l_sptr ); \
    return reinterpret_cast< vital_landmark_t* >( l_sptr.get() ); \
  ); \
  return 0; \
} \
\
/** Create a new default instance of a landmark */ \
vital_landmark_t* \
vital_landmark_##S##_new_default( vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_new_default", eh, \
    auto l_sptr = std::make_shared< vital::landmark_<T> >(); \
    return reinterpret_cast< vital_landmark_t* >( l_sptr.get() ); \
  ); \
  return 0;\
} \
\
/** Set 3D location of a landmark instance */ \
void \
vital_landmark_##S##_set_loc( vital_landmark_t *l, \
                              vital_eigen_matrix3x1##S##_t const *loc, \
                              vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix<T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_loc", eh, \
    REINTERP_TYPE( matrix_t const, loc, loc_ptr ); \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    l_ptr->set_loc( *loc_ptr ); \
  ); \
} \
\
/** Set the scale of the landmark */ \
void \
vital_landmark_##S##_set_scale( vital_landmark_t *l, T scale, \
                                vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_scale", eh, \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    l_ptr->set_scale( scale );\
  ); \
} \
\
/** Set the normal vector of the landmark */ \
void \
vital_landmark_##S##_set_normal( vital_landmark_t *l, \
                                 vital_eigen_matrix3x1##S##_t const *normal, \
                                 vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix<T, 3, 1> matrix_t; \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_normal", eh, \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    REINTERP_TYPE( matrix_t const, normal, n_ptr ); \
    l_ptr->set_normal( *n_ptr ); \
  ); \
} \
\
/** Set the covariance of the landmark */ \
void \
vital_landmark_##S##_set_covar( vital_landmark_t *l, \
                                vital_covariance_3##S##_t const *covar, \
                                vital_error_handle_t *eh ) \
{ \
  typedef vital::covariance_<3, T> covar_t; \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_covar", eh, \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    REINTERP_TYPE( covar_t const, covar, covar_ptr ); \
    l_ptr->set_covar( *covar_ptr ); \
  ); \
} \
\
/** Set the color of the landmark */ \
void \
vital_landmark_##S##_set_color( vital_landmark_t *l, \
                                vital_rgb_color_t const *c, \
                                vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_color", eh, \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    REINTERP_TYPE( vital::rgb_color const, c, c_ptr ); \
    l_ptr->set_color( *c_ptr ); \
  ); \
} \
\
/** Set the observations of the landmark */ \
void \
vital_landmark_##S##_set_observations( vital_landmark_t *l, \
                                       unsigned observations, \
                                       vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_landmark_" #S "_set_observations", eh, \
    TRY_DYNAMIC_CAST( vital::landmark_<T>, \
                      vital_c::LANDMARK_SPTR_CACHE.get( l ).get(), \
                      l_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast to '" #S "' type for data " \
                          "access method." ); \
      return; \
    } \
    l_ptr->set_observations( observations ); \
  ); \
}


DEFINE_TYPED_OPERATIONS( double, d )
DEFINE_TYPED_OPERATIONS( float,  f )

#undef DEFINE_TYPED_OPERATIONS
