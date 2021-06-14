// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::detected_object classes
 */

#include "detected_object.h"

#include <vital/types/detected_object.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/detected_object.h>
#include <vital/bindings/c/helpers/detected_object_type.h>
#include <vital/bindings/c/types/detected_object_type.h>

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
    "vital_detected_object_new_with_bbox", 0,;
    kwiver::vital::detected_object_type_sptr dot_sptr;

    if( dot != NULL )
    {
      dot_sptr = kwiver::vital_c::DOT_SPTR_CACHE.get( dot );
    }

    kwiver::vital::bounding_box_d& bbox_ref =
      *reinterpret_cast< kwiver::vital::bounding_box_d* >( bbox );

    kwiver::vital::detected_object_sptr do_sptr =
      std::make_shared< kwiver::vital::detected_object >( bbox_ref, confidence, dot_sptr );

    kwiver::vital_c::DOBJ_SPTR_CACHE.store( do_sptr );
    return reinterpret_cast<vital_detected_object_t*>( do_sptr.get() );
  );
  return 0;
}

vital_detected_object_t* vital_detected_object_copy(vital_detected_object_t * obj)
{
  STANDARD_CATCH(
    "vital_detected_object_copy", 0,;
    kwiver::vital::detected_object_sptr do_sptr =
      std::make_shared< kwiver::vital::detected_object >(
        *reinterpret_cast<kwiver::vital::detected_object*>( obj ) );
    kwiver::vital_c::DOBJ_SPTR_CACHE.store( do_sptr );
    return reinterpret_cast<vital_detected_object_t*>( do_sptr.get() );
  );
  return 0;
}

void vital_detected_object_destroy( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "vital_detected_object_destroy", 0,
    kwiver::vital_c::DOBJ_SPTR_CACHE.erase( obj );
  );
}

vital_bounding_box_t* vital_detected_object_bounding_box( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "vital_detected_object_bounding_box", 0,
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
    "vital_detected_object_confidence", 0,
    return kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->confidence();
  );
  return 0;
}

void vital_detected_object_set_confidence( vital_detected_object_t * obj,
                                           double conf )
{
  STANDARD_CATCH(
    "vital_detected_object_set_confidence", 0,
    kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_confidence( conf );
  );
}

vital_detected_object_type_t* vital_detected_object_get_type( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "vital_detected_object_get_type", 0,
    auto dot = kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->type();
    return reinterpret_cast<vital_detected_object_type_t*>( dot.get() );
  );
  return 0;
}

void vital_detected_object_set_type( vital_detected_object_t *      obj,
                                     vital_detected_object_type_t * dot )
{
  STANDARD_CATCH(
    "vital_detected_object_set_type", 0,
    auto ldot = std::make_shared< kwiver::vital::detected_object_type > (
      * reinterpret_cast< kwiver::vital::detected_object_type* >(dot) );
    //+ dot is managed by sptr
    kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_type( ldot );
  );
}

int64_t vital_detected_object_index( vital_detected_object_t * obj )
{
  STANDARD_CATCH(
    "vital_detected_object_index", 0,
    return kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->index();
  );
  return 0;
}

void vital_detected_object_set_index(vital_detected_object_t * obj,
                                     int64_t idx)
{
  kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_index(idx);
}

char* vital_detected_object_detector_name(vital_detected_object_t * obj)
{
  std::string sname =  kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->detector_name();
  //+ TBD need to return pointer to persistent string
  return 0;
}

void vital_detected_object_detector_set_name(vital_detected_object_t * obj,
                                            char* name )
{
  std::string sname(name);
  kwiver::vital_c::DOBJ_SPTR_CACHE.get( obj )->set_detector_name( sname );
}

vital_image_t* vital_detected_object_mask(VITAL_UNUSED vital_detected_object_t * obj)
{
  //+ TBD need to look up image_sptr in cache
  return 0;
}

void vital_detected_object_set_mask(VITAL_UNUSED vital_detected_object_t * obj,
                                    VITAL_UNUSED vital_image_t* mask)
{
  //+ TBD need to look up image in cache
}
