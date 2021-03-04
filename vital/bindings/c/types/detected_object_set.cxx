// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

typedef std::vector< kwiver::vital::detected_object_sptr > vector_t;

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
    "vital_detected_object_set_from_sptr", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( sptr );
    return reinterpret_cast<vital_detected_object_set_t*>( sptr.get() );
    );
  return 0;
}

vital_detected_object_set_t* vital_detected_object_set_from_c_pointer( kwiver::vital::detected_object_set* ptr )
{
  STANDARD_CATCH(
    "vital_detected_object_set_from_c_pointer", 0,
    kwiver::vital::detected_object_set_sptr sptr(ptr);
    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.store( sptr );
    return reinterpret_cast<vital_detected_object_set_t*>( ptr );
    );
  return 0;
}

kwiver::vital::detected_object_set_sptr vital_detected_object_set_to_sptr( vital_detected_object_set_t* handle )
{
  STANDARD_CATCH(
    "vital_detected_object_set_to_sptr", 0,

    return kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( handle );
    );
  return kwiver::vital::detected_object_set_sptr();
}

// ------------------------------------------------------------------
vital_detected_object_set_t* vital_detected_object_set_new()
{
  STANDARD_CATCH(
    "vital_detected_object_set_new", 0,

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
    "vital_detected_object_set_new_from_list", 0,

    vector_t input( n );
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
    "vital_detected_object_set_destroy", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.erase( obj );
  );

}

// ------------------------------------------------------------------
void vital_detected_object_set_add( vital_detected_object_set_t* set,
                                    vital_detected_object_t* obj )
{
  STANDARD_CATCH(
    "vital_detected_object_set_add", 0,

    kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( set )->add( kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj ) );
    );
}

// ------------------------------------------------------------------
size_t vital_detected_object_set_size( vital_detected_object_set_t* obj)
{
  STANDARD_CATCH(
    "vital_detected_object_set_size", 0,

    return kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->size();
    );
  return 0;
}

// ------------------------------------------------------------------
void vital_detected_object_set_select_threshold( vital_detected_object_set_t* obj,
                                                 double thresh,
                                                 vital_detected_object_t*** output,
                                                 size_t* length )
{
  STANDARD_CATCH(
    "vital_detected_object_set_select_threshold", 0,

    auto sel_set = kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->select( thresh );

    // select to get vector
    *output = (vital_detected_object_t**) malloc( sizeof( vital_detected_object_t* ) * sel_set->size() );
    *length = sel_set->size();

    auto ie = sel_set->cend();
    size_t i = 0;
    for ( auto ix = sel_set->cbegin(); ix != ie; ++ix )
    {
      kwiver::vital_c::DOBJ_SPTR_CACHE.store( *ix );
      (*output)[i] = reinterpret_cast< vital_detected_object_t* >( (*ix).get() );
    }
    );
}

// ------------------------------------------------------------------
void vital_detected_object_set_select_class_threshold( vital_detected_object_set_t* obj,
                                                       const char* class_name,
                                                       double thresh,
                                                       vital_detected_object_t*** output,
                                                       size_t* length )
{
  STANDARD_CATCH(
    "vital_detected_object_set_select_class_threshold", 0,

    auto sel_set = kwiver::vital_c::DOBJ_SET_SPTR_CACHE.get( obj )->select( std::string (class_name), thresh );

    // select to get vector
    *output = (vital_detected_object_t**) malloc( sizeof( vital_detected_object_t* ) * sel_set->size() );
    *length = sel_set->size();

    auto ie = sel_set->cend();
    size_t i = 0;
    for ( auto ix = sel_set->cbegin(); ix != ie; ++ix )
    {
      kwiver::vital_c::DOBJ_SPTR_CACHE.store( *ix );
      (*output)[i] = reinterpret_cast< vital_detected_object_t* >( (*ix).get() );
    }
  );
}
