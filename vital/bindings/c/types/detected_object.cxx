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
 * \brief C interface to vital::detected_object classes
 */

#include "detected_object.h"

#include <vital/types/detected_object.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/detected_object.h>
#include <vital/bindings/c/helpers/detected_object_type.h>

#include <memory>

namespace kwiver {
namespace vital_c {

// Allocate our shared pointer cache object
SharedPointerCache< kwiver::vital::detected_object, vital_detected_object_t >
  DOBJ_SPTR_CACHE( "detected_object" );

} }


vital_detected_object_t* vital_detected_object_new_with_bbox( vital_bounding_box_t* bbox,
                                                              double confidence,
                                                              vital_detected_object_type_t* dot)
{
  STANDARD_CATCH(
    "C::detected_object:new", 0,
    auto dotsp = kwiver::vital_c::DOT_SPTR_CACHE.get( dot );
    kwiver::vital::bounding_box_d& bbox_ref =
    * reinterpret_cast< kwiver::vital::bounding_box_d* >( bbox );

    auto det_obj_sptr = std::make_shared< kwiver::vital::detected_object> (
      bbox_ref, confidence, dotsp );

    kwiver::vital_c::DOBJ_SPTR_CACHE.store( det_obj_sptr );

    return reinterpret_cast<vital_detected_object_t*>( det_obj_sptr.get() );
  );
  return 0;
}


void vital_detected_object_destroy( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "C::detected_object::destroy", 0,
    kwiver::vital_c::DOBJ_SPTR_CACHE.erase( obj );
  );
}


vital_bounding_box_t* vital_detected_object_bounding_box( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "C::detected_object::bounding_box", 0,
    kwiver::vital::bounding_box_d* bbox = new kwiver::vital::bounding_box_d(
      kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->bounding_box() );
    return reinterpret_cast<vital_bounding_box_t*>( bbox );
  );
  return 0;
}


void vital_detected_object_set_bounding_box( vital_detected_object_t * obj,
                                             vital_bounding_box_t* bbox )
{
  kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_bounding_box(
    *reinterpret_cast<kwiver::vital::bounding_box_d*>( bbox ) );
}


double vital_detected_object_confidence( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "C::detected_object::confidence", 0,
    return kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->confidence();
  );
  return 0;
}


void vital_detected_object_set_confidence( vital_detected_object_t * obj,
                                           double conf )
{
  STANDARD_CATCH(
    "C::detected_object::set_confidence", 0,
    kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_confidence( conf );
  );
}


vital_detected_object_type_t* vital_detected_object_get_type( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "C::detected_object::object_type", 0,
    auto dot = kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->type();
    return reinterpret_cast<vital_detected_object_type_t*>( dot.get() );
  );
  return 0;
}


void vital_detected_object_set_type( vital_detected_object_t *      obj,
                                     vital_detected_object_type_t * dot )
{
  STANDARD_CATCH(
    "C::detected_object::set_type", 0,
    auto ldot = std::make_shared< kwiver::vital::detected_object_type > (
      * reinterpret_cast< kwiver::vital::detected_object_type* >(dot) );
    //+ DOT is managed by sptr
    kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_type( ldot );
  );
}
