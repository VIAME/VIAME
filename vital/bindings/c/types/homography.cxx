/*ckwg +29
 * Copyright 2016, 2019 by Kitware, Inc.
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

#include "homography.h"

#include <vital/exceptions/math.h>
#include <vital/types/homography.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/homography.h>


namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::vital::homography, vital_homography_t >
  HOMOGRAPHY_SPTR_CACHE( "homography" );

}
}


using namespace kwiver;


/// Destroy a homography instance
void
vital_homography_destroy( vital_homography_t *h, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_homography_destroy", eh,
    vital_c::HOMOGRAPHY_SPTR_CACHE.erase( h );
  );
}


/// Get the name of the descriptor instance's underlying data type
char const*
vital_homography_type_name( vital_homography_t const *h,
                            vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_homography_type_name", eh,
    auto h_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( h );
    return h_sptr->data_type().name();
  );
  return 0;
}


/// Create a clone of a homography instance
vital_homography_t*
vital_homography_clone( vital_homography_t const *h, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_homography_clone", eh,
    auto h_clone_sptr =
      std::static_pointer_cast< kwiver::vital::homography >(
        vital_c::HOMOGRAPHY_SPTR_CACHE.get( h )->clone() );
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( h_clone_sptr );
    return reinterpret_cast< vital_homography_t* >( h_clone_sptr.get() );
  );
  return 0;
}


/// Get a new homography instance that is normalized
vital_homography_t*
vital_homography_normalize( vital_homography_t const *h,
                            vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_homography_normalize", eh,
    auto h_norm_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( h )->normalize();
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( h_norm_sptr );
    return reinterpret_cast< vital_homography_t* >( h_norm_sptr.get() );
  );
  return 0;
}


/// Get a new homography instance that has been inverted
vital_homography_t*
vital_homography_inverse( vital_homography_t const *h,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_homography_inverse", eh,
    auto h_inv_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( h )->inverse();
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( h_inv_sptr );
    return reinterpret_cast< vital_homography_t* >( h_inv_sptr.get() );
  );
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
// Type specific functions and constructors

#define DEFINE_TYPED_OPERATIONS( T, S ) \
\
/* New identity homography */ \
vital_homography_t* \
vital_homography_##S##_new_identity( vital_error_handle_t *eh ) \
{ \
  typedef vital::homography_< T > homog_t; \
  STANDARD_CATCH( \
    "vital_homography_" #S "_new_identity", eh, \
    auto h_sptr = std::make_shared< homog_t >(); \
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( h_sptr ); \
    return reinterpret_cast< vital_homography_t* >( h_sptr.get() ); \
  ); \
  return 0; \
} \
\
/* New homography from a provided transformation matrix */ \
vital_homography_t* \
vital_homography_##S##_new_from_matrix( vital_eigen_matrix3x3##S##_t const *m, \
                                        vital_error_handle_t *eh ) \
{ \
  typedef vital::homography_< T > homog_t; \
  STANDARD_CATCH( \
    "vital_homography_" #S "_new_from_matrix", eh, \
    REINTERP_TYPE( homog_t::matrix_t const, m, m_ptr ); \
    auto h_sptr = std::make_shared< homog_t >( *m_ptr ); \
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( h_sptr ); \
    return reinterpret_cast< vital_homography_t* >( h_sptr.get() ); \
  ); \
  return 0; \
} \
\
/* Get a homography's transformation matrix */ \
vital_eigen_matrix3x3##S##_t* \
vital_homography_##S##_as_matrix( vital_homography_t const *h, \
                                  vital_error_handle_t *eh ) \
{ \
  typedef vital::homography_< T > homog_t; \
  STANDARD_CATCH( \
    "vital_homography_" #S "_as_matrix", eh, \
    auto h_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( h ); \
    TRY_DYNAMIC_CAST( homog_t const, h_sptr.get(), h_ptr ) \
    { \
      throw "Failed to cast homography to '" #S "' type instance."; \
    } \
    homog_t::matrix_t *m = new homog_t::matrix_t( h_ptr->get_matrix() ); \
    return reinterpret_cast< vital_eigen_matrix3x3##S##_t* >( m ); \
  ); \
  return 0; \
} \
\
/* Map a 2D point using this homography */ \
vital_eigen_matrix2x1##S##_t* \
vital_homography_##S##_map_point( vital_homography_t const *h, \
                                  vital_eigen_matrix2x1##S##_t const *p, \
                                  vital_error_handle_t *eh ) \
{ \
  typedef vital::homography_< T > homog_t; \
  typedef Eigen::Matrix< T, 2, 1 > point_t; \
  STANDARD_CATCH( \
    "vital_homography_" #S "_map_point", eh, \
    auto h_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( h ); \
    TRY_DYNAMIC_CAST( homog_t const, h_sptr.get(), h_ptr ) \
    { \
      throw "Failed to cast homography to '" #S "' type instance."; \
    } \
    REINTERP_TYPE( point_t const, p, p_ptr ); \
    point_t *q_ptr = NULL; \
    try \
    { \
      q_ptr = new point_t( h_ptr->map_point( *p_ptr ) ); \
    } \
    catch( vital::point_maps_to_infinity const&e ) \
    { \
      POPULATE_EH( eh, 1, e.what() ); \
    } \
    return reinterpret_cast< vital_eigen_matrix2x1##S##_t* >( q_ptr ); \
  ); \
  return 0; \
} \
\
/* Multiply one homography against another, returning a new product homography */ \
vital_homography_t* \
vital_homography_##S##_multiply( vital_homography_t const *lhs, \
                                 vital_homography_t const *rhs, \
                                 vital_error_handle_t *eh ) \
{ \
  typedef vital::homography_< T > homog_t; \
  STANDARD_CATCH( \
    "vital_homography_" #S "_multiply", eh, \
    auto lhs_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( lhs ); \
    auto rhs_sptr = vital_c::HOMOGRAPHY_SPTR_CACHE.get( rhs ); \
    TRY_DYNAMIC_CAST( homog_t const, lhs_sptr.get(), lhs_ptr ) \
    { \
      throw "Failed to cast left-hand homography to '" #S "' type instance."; \
    } \
    TRY_DYNAMIC_CAST( homog_t const, rhs_sptr.get(), rhs_ptr ) \
    { \
      throw "Failed to cast right-hand homography to '" #S "' type instance."; \
    } \
    auto p_sptr = std::make_shared< homog_t >( (*lhs_ptr) * (*rhs_ptr) ); \
    vital_c::HOMOGRAPHY_SPTR_CACHE.store( p_sptr ); \
    return reinterpret_cast< vital_homography_t* >( p_sptr.get() ); \
  ); \
  return 0; \
}


DEFINE_TYPED_OPERATIONS( double, d )
DEFINE_TYPED_OPERATIONS( float,  f )
