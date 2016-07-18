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
 * \brief Implementation of C interface to vital::feature
 */

#include "feature.h"

#include <vital/bindings/c/helpers/feature.h>
#include <vital/types/feature.h>

#include <vital/types/color.h>


namespace kwiver {
namespace vital_c {

SharedPointerCache< vital::feature, vital_feature_t >
  FEATURE_SPTR_CACHE( "feature" );

}
}


using namespace kwiver;


/// Destroy a feature instance
void
vital_feature_destroy( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_destroy", eh,
    vital_c::FEATURE_SPTR_CACHE.erase( f );
  );
}


/// Get 2D image location
vital_eigen_matrix2x1d_t*
vital_feature_loc( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_loc", eh,
    // Copy location vector onto heap as new instance
    return reinterpret_cast< vital_eigen_matrix2x1d_t* >(
      new vital::vector_2d( vital_c::FEATURE_SPTR_CACHE.get( f )->loc() )
    );
  );
  return 0;
}


/// Get feature magnitude
double
vital_feature_magnitude( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_magnitude", eh,
    return vital_c::FEATURE_SPTR_CACHE.get( f )->magnitude();
  );
  return 0;
}


/// Get feature scale
double
vital_feature_scale( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_scale", eh,
    return vital_c::FEATURE_SPTR_CACHE.get( f )->scale();
  );
  return 0;
}


/// Get feature angle
double
vital_feature_angle( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_angle", eh,
    return vital_c::FEATURE_SPTR_CACHE.get( f )->angle();
  );
  return 0;
}


/// Get feature 2D covariance
vital_covariance_2d_t*
vital_feature_covar( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_covar", eh,
    // Copy covar onto heap as new instance
    return reinterpret_cast< vital_covariance_2d_t* >(
      new vital::covariance_2d( vital_c::FEATURE_SPTR_CACHE.get( f )->covar() )
    );
  );
  return 0;
}


/// Get Feature location's pixel color
vital_rgb_color_t*
vital_feature_color( vital_feature_t *f, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_color", eh,
    return reinterpret_cast< vital_rgb_color_t* >(
      new vital::rgb_color( vital_c::FEATURE_SPTR_CACHE.get( f )->color() )
    );
  );
  return 0;
}


/// Define type-specific feature functions
/**
 * \param T data type
 * \param S standard data type character symbol
 */
#define DEFINE_FEATURE_OPERATIONS( T, S ) \
\
/** Create a new typed feature instance. */ \
vital_feature_t* \
vital_feature_##S##_new( vital_eigen_matrix2x1##S##_t *loc, T mag, T scale, \
                         T angle, vital_rgb_color_t *color, \
                         vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix< T, 2, 1 > cpp_matrix_t; \
  STANDARD_CATCH( \
    "vital_feature_" #S "_new", eh, \
    REINTERP_TYPE( cpp_matrix_t, loc, loc_ptr ); \
    REINTERP_TYPE( vital::rgb_color, color, color_ptr ); \
    auto f_sptr = std::make_shared< vital::feature_<T> >( \
      *loc_ptr, mag, scale, angle, *color_ptr \
    ); \
    vital_c::FEATURE_SPTR_CACHE.store( f_sptr ); \
    return reinterpret_cast< vital_feature_t* >( f_sptr.get() ); \
  ); \
  return 0; \
} \
\
/** Create a new typed feature instance with default parameters */ \
vital_feature_t* \
vital_faeture_##S##_new_default( vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_feature_" #S "_new_default", eh, \
    auto f_sptr = std::make_shared< vital::feature_<T> >(); \
    vital_c::FEATURE_SPTR_CACHE.store( f_sptr ); \
    return reinterpret_cast< vital_feature_t* >( f_sptr.get() ); \
  ); \
  return 0; \
} \
\
/** Set feature location vector */ \
void \
vital_feature_##S##_set_loc( vital_feature_t *f, \
                             vital_eigen_matrix2x1##S##_t *l, \
                             vital_error_handle_t *eh ) \
{ \
  typedef Eigen::Matrix<T, 2, 1> cpp_matrix_t; \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_loc", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    REINTERP_TYPE( cpp_matrix_t, l, l_ptr ); \
    f_ptr->set_loc( *l_ptr ); \
  ); \
} \
\
/** Set feature magnitude */ \
void \
vital_feature_##S##_set_magnitude( vital_feature_t *f, \
                                   T mag, \
                                   vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_magnitude", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    f_ptr->set_magnitude( mag ); \
  ); \
} \
\
/** Set feature scale */ \
void \
vital_feature_##S##_set_scale( vital_feature_t *f, \
                               T scale, \
                               vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_scale", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    f_ptr->set_scale( scale ); \
  ); \
} \
\
/** Set feature angle */ \
void \
vital_feature_##S##_set_angle( vital_feature_t *f, \
                               T angle, \
                               vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_angle", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    f_ptr->set_angle( angle ); \
  ); \
} \
\
/** Set feature covariance matrix */ \
void \
vital_feature_##S##_set_covar( vital_feature_t *f, \
                               vital_covariance_2##S##_t *covar, \
                               vital_error_handle_t *eh ) \
{ \
  typedef vital::covariance_< 2, T > cpp_covar_t; \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_covar", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    REINTERP_TYPE( cpp_covar_t, covar, covar_ptr ); \
    f_ptr->set_covar( *covar_ptr ); \
  ); \
} \
\
/** Set feature color */ \
void \
vital_feature_##S##_set_color( vital_feature_t *f, \
                               vital_rgb_color_t *c, \
                               vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_feature_" #S "_set_color", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return; \
    } \
    REINTERP_TYPE( vital::rgb_color, c, c_ptr ); \
    f_ptr->set_color( *c_ptr ); \
  ); \
} \
\
/** Get the name of the instance's data type */ \
char const* \
vital_feature_##S##_type_name( vital_feature_t *f, \
                               vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_featur_" #S "_type_name", eh, \
    auto f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f ); \
    TRY_DYNAMIC_CAST( vital::feature_<T>, f_sptr.get(), f_ptr ) \
    { \
      POPULATE_EH( eh, 1, "Failed dynamic cast to '" #S "' type for data " \
                          "access." ); \
      return 0; \
    } \
    return f_ptr->data_type().name(); \
  ); \
  return 0; \
}


DEFINE_FEATURE_OPERATIONS( double, d )
DEFINE_FEATURE_OPERATIONS( float,  f )
