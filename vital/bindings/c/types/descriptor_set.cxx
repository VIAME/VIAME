/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief Implementation of vital::descriptor_set interface
 */

#include "descriptor_set.h"
#include "descriptor_set.hxx"

#include <vector>

#include <vital/bindings/c/helpers/descriptor.h>
#include <vital/bindings/c/helpers/descriptor_set.h>


namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::vital::descriptor_set, vital_descriptor_set_t >
  DESCRIPTOR_SET_SPTR_CACHE( "descriptor_set" );

}
}

using namespace kwiver;


/// Create a new descriptor set from the array of descriptors.
vital_descriptor_set_t*
vital_descriptor_set_new( vital_descriptor_t const **d_array,
                          size_t d_array_length,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_new", eh,

    // Convert the input array of descriptors into a std::vector of
    // vital::descriptor instances.
    std::vector< vital::descriptor_sptr > descriptor_sptr_vec;
    for ( size_t i = 0; i < d_array_length; ++i )
    {
      descriptor_sptr_vec.push_back(
        kwiver::vital_c::DESCRIPTOR_SPTR_CACHE.get( d_array[i] )
      );
    }

    // Create a new simple_descriptor_set using the given descriptors, if any.
    auto ds_sptr =
      std::make_shared< vital::simple_descriptor_set >( descriptor_sptr_vec );
    vital_c::DESCRIPTOR_SET_SPTR_CACHE.store( ds_sptr );
    return reinterpret_cast< vital_descriptor_set_t* >( ds_sptr.get() );
  );
  return NULL;
}


/// Create a vital_descriptor_set_t around an existing shared pointer.
vital_descriptor_set_t*
vital_descriptor_set_new_from_sptr( kwiver::vital::descriptor_set_sptr ds_sptr,
                                    vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_new_from_sptr", eh,
    // Store the shared pointer in our cache and return the handle.
    vital_c::DESCRIPTOR_SET_SPTR_CACHE.store( ds_sptr );
    return reinterpret_cast< vital_descriptor_set_t* >( ds_sptr.get() );
  );
  return NULL;
}


/// Destroy a descriptor set
void
vital_descriptor_set_destroy( vital_descriptor_set_t const *ds,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_destroy", eh,
    vital_c::DESCRIPTOR_SET_SPTR_CACHE.erase( ds );
  );
}


/// Get the size of a descriptor set
size_t
vital_descriptor_set_size( vital_descriptor_set_t const *ds,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_size", eh,
    return vital_c::DESCRIPTOR_SET_SPTR_CACHE.get( ds )->size();
  );
  return 0;
}


/// Get the descriptors stored in this set.
void
vital_descriptor_set_get_descriptors( vital_descriptor_set_t const *ds,
                                      vital_descriptor_t ***out_d_array,
                                      size_t *out_d_array_length,
                                      vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_get_descriptors", eh,
    // Get vector of descriptor sptrs from c++ set instance.
    vital::descriptor_set_sptr ds_sptr = vital_c::DESCRIPTOR_SET_SPTR_CACHE.get( ds );
    // Store descriptor sptrs in cache structure and store pointers in
    // malloc'd array.
    vital_descriptor_t **d_array = (vital_descriptor_t**)malloc(
      sizeof(vital_descriptor_t*) * ds_sptr->size()
    );
    if ( d_array == NULL )
    {
      throw "Failed to allocate memory for descriptor handle array.";
    }
    for( size_t i = 0; i < ds_sptr->size(); ++i )
    {
      vital_c::DESCRIPTOR_SPTR_CACHE.store( ds_sptr->at(i) );
      d_array[i] = reinterpret_cast< vital_descriptor_t* >( ds_sptr->at(i).get());
    }
    *out_d_array = d_array;
    *out_d_array_length = ds_sptr->size();
  );
}


/// Get the vital::descriptor_set shared pointer for a handle.
kwiver::vital::descriptor_set_sptr
vital_descriptor_set_to_sptr( vital_descriptor_set_t* ds,
                              vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "vital_descriptor_set_to_sptr", eh,
    // Return the cached shared pointer.
    return vital_c::DESCRIPTOR_SET_SPTR_CACHE.get( ds );
  );
  return kwiver::vital::descriptor_set_sptr();
}
