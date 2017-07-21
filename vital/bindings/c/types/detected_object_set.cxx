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
 * \brief Interface for detected_object_set class
 */

#include "detected_object_set.h"
#include "detected_object_set.hxx"

#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/detected_object.h>

#include <memory>

namespace kwiver {
namespace vital_c {

// Allocate our shared pointer cache object
SharedPointerCache< kwiver::vital::detected_object_set, vital_detected_object_set_t >
  DOBJ_SET_SPTR_CACHE( "detected_object_set" );

} }

// ==================================================================
// These two functions support C++ access to the SPTR_CACHE.

/**
 * @brief Accept shared pointer to detected object set.
 *
 * This function takes a pointer to a shared_pointer and adds it to
 * the SPTR_CACHE in the same way as a constructor (above). This
 * allows us to manage an already existing object.
 *
 * @param sptr Pointer to shared pointer
 *
 * @return Opaque object pointer/handle
 */
vital_detected_object_set_t* vital_detected_object_set_from_sptr( kwiver::vital::detected_object_set_sptr sptr )
{
  STANDARD_CATCH(
    "C::detected_object_set::from_sptr", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( sptr );
    return reinterpret_cast<vital_detected_object_set_t*>( sptr.get() );
    );
  return 0;
}


vital_detected_object_set_t* vital_detected_object_set_from_c_pointer( kwiver::vital::detected_object_set* ptr )
{
  STANDARD_CATCH(
    "C::detected_object_set::from_c_ptr", 0,
    kwiver::vital::detected_object_set_sptr sptr(ptr);
    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( sptr );
    return reinterpret_cast<vital_detected_object_set_t*>( ptr );
    );
  return 0;
}


kwiver::vital::detected_object_set_sptr vital_detected_object_set_to_sptr( vital_detected_object_set_t* handle )
{
  STANDARD_CATCH(
    "C::detected_object_set::to_sptr", 0,

    return kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( handle );
    );
  return kwiver::vital::detected_object_set_sptr();
}


// ------------------------------------------------------------------
vital_detected_object_set_t* vital_detected_object_set_new()
{
  STANDARD_CATCH(
    "C::detected_object_set:new", 0,

    auto dot_sptr = std::make_shared< kwiver::vital::detected_object_set> ();

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( dot_sptr );
    return reinterpret_cast<vital_detected_object_set_t*>( dot_sptr.get() );
  );
  return 0;

}


// ------------------------------------------------------------------
vital_detected_object_set_t*
vital_detected_object_set_new_from_list( vital_detected_object_t**  dobj,
                                         size_t                     n )
{
  STANDARD_CATCH(
    "C::detected_object_set:new_from_list", 0,

    kwiver::vital::detected_object::vector_t input( n );
    for ( size_t i = 0; i < n; ++i )
    {
      input.push_back( kwiver::vital_c::DOBJ_SPTR_CACHE.get( dobj[i] ) );
    }

    auto obj_set = std::make_shared< kwiver::vital::detected_object_set > ( input );

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( obj_set );
    return reinterpret_cast< vital_detected_object_set_t* > ( obj_set.get() );
    );
  return 0;
}


// ------------------------------------------------------------------
void vital_detected_object_set_destroy( vital_detected_object_set_t* obj)
{
  STANDARD_CATCH(
    "C::detected_object_set::destroy", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.erase( obj );
  );

}


// ------------------------------------------------------------------
void vital_detected_object_set_add( vital_detected_object_set_t* set,
                                    vital_detected_object_t* obj )
{
  STANDARD_CATCH(
    "C::detected_object_set::add", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( set )->add( kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj ) );
    );
}


// ------------------------------------------------------------------
size_t vital_detected_object_set_size( vital_detected_object_set_t* obj)
{
  STANDARD_CATCH(
    "C::detected_object_set::size", 0,

    return kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->size();
    );
  return 0;
}


// ------------------------------------------------------------------
vital_detected_object_t** vital_detected_object_set_select_threshold( vital_detected_object_set_t* obj,
                                                                      double thresh,
                                                                      size_t* length )
{
  STANDARD_CATCH(
    "C::detected_object_set::select_threshold", 0,

    auto sel_set = kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->select( thresh );

    // select to get vector
    vital_detected_object_t** output_set =
      (vital_detected_object_t**) calloc( sizeof( vital_detected_object_t* ), sel_set.size() );

    *length = sel_set.size();

    for ( size_t i = 0; i < sel_set.size(); ++i )
    {
      output_set[i] = reinterpret_cast< vital_detected_object_t* >( sel_set[i].get() );
    }
    return output_set;
    );
  return 0;
}


// ------------------------------------------------------------------
vital_detected_object_t** vital_detected_object_set_select_class_threshold( vital_detected_object_set_t* obj,
                                                                            const char* class_name,
                                                                            double thresh,
                                                                            size_t* length )
{
  STANDARD_CATCH(
    "C::detected_object_set::select_class_threshold", 0,

    auto sel_set = kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->select( std::string (class_name), thresh );

    // select to get vector
    vital_detected_object_t** output_set =
      (vital_detected_object_t**) calloc( sizeof( vital_detected_object_t* ), sel_set.size() );

    *length = sel_set.size();

    for (size_t i = 0; i < sel_set.size(); ++i )
    {
      output_set[i] = reinterpret_cast< vital_detected_object_t* >( sel_set[i].get() );
    }
    return output_set;
    );
  return 0;
}
