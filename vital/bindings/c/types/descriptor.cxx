// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of vital::descriptor interface
 */

#include "descriptor.h"

#include <vector>

#include <vital/bindings/c/helpers/descriptor.h>

namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::vital::descriptor, vital_descriptor_t >
  DESCRIPTOR_SPTR_CACHE( "descriptor" );

}
}

using namespace kwiver;

/// Destroy a descriptor instance
void
vital_descriptor_destroy( vital_descriptor_t const *d,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_destroy", eh,
    vital_c::DESCRIPTOR_SPTR_CACHE.erase( d );
  );
}

/// Get the number of elements in the descriptor
size_t
vital_descriptor_size( vital_descriptor_t const *d,
                       vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_size", eh,
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    return d_sptr->size();
  );
  return 0;
}

/// Get the number of bytes used to represent the descriptor's data
size_t
vital_descriptor_num_bytes( vital_descriptor_t const *d,
                            vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_num_bytes", eh,
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    return d_sptr->num_bytes();
  );
  return 0;
}

/// Convert the descriptor into a new array of bytes
unsigned char*
vital_descriptor_as_bytes( vital_descriptor_t const *d,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_as_bytes", eh,
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    auto v = d_sptr->as_bytes();
    unsigned char *v_ptr = (unsigned char*)malloc(sizeof(unsigned char) * d_sptr->num_bytes());
    for( size_t i=0; i < d_sptr->num_bytes(); ++i )
    {
      v_ptr[i] = v[i];
    }
    return v_ptr;
  );
  return 0;
}

/// Convert the descriptor into a new array of doubles
double*
vital_descriptor_as_doubles( vital_descriptor_t const *d,
                             vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_as_doubles", eh,
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    std::vector<double> v = d_sptr->as_double();
    double *v_ptr = (double*)malloc(sizeof(double) * v.size());
    for( size_t i=0; i < v.size(); ++i )
    {
      v_ptr[i] = v[i];
    }
    return v_ptr;
  );
  return 0;
}

/// Get the name of the descriptor instance's data type
char const*
vital_descriptor_type_name( vital_descriptor_t const *d,
                            vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_type_name", eh,
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    return d_sptr->data_type().name();
  );
  return 0;
}

#define DEFINE_TYPED_OPERATIONS( T, S ) \
\
/* Create a new descriptor of a the given size (vital::descriptor_dynamic<T>) */ \
vital_descriptor_t* \
vital_descriptor_new_##S( size_t size, \
                          vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_descriptor_new_" #S, eh, \
    auto d_sptr = std::make_shared< vital::descriptor_dynamic<T> >( size ); \
    vital_c::DESCRIPTOR_SPTR_CACHE.store( d_sptr ); \
    return reinterpret_cast< vital_descriptor_t* >( d_sptr.get() ); \
  ); \
  return 0; \
} \
\
/* Get the pointer to the descriptor's data array */ \
T* \
vital_descriptor_get_##S##_raw_data( vital_descriptor_t *d, \
                                     vital_error_handle_t *eh ) \
{ \
  STANDARD_CATCH( \
    "vital_descriptor_" #S "_raw_data", eh, \
    auto d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d ); \
    TRY_DYNAMIC_CAST( vital::descriptor_dynamic<T>, d_sptr.get(), d_dyn ) \
    { \
      POPULATE_EH( eh, 1, "Failed to dynamic cast descriptor to " #S " type " \
                          "for data access" ); \
      return 0; \
    } \
    /* Succeeded cast at this point */ \
    return d_dyn->raw_data(); \
  ); \
  return 0; \
}

DEFINE_TYPED_OPERATIONS( double, d )
DEFINE_TYPED_OPERATIONS( float,  f )

#undef DEFINE_TYPED_OPERATIONS
