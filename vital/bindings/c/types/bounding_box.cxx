// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::bounding_box C interface implementation
 */

#include "bounding_box.h"

#include <vital/types/bounding_box.h>

#include <vital/bindings/c/helpers/c_utils.h>

typedef kwiver::vital::bounding_box_d::vector_type point_t;

// ------------------------------------------------------------------
vital_bounding_box_t* vital_bounding_box_new_from_vectors( double* ul, double* lr)
{
  // Convert arrays to local vector tytpe
  const point_t ulp( ul[0], ul[1] );
  const point_t lrp( lr[0], lr[1] );

  STANDARD_CATCH(
    "vital_bounding_box_new_from_vectors", 0,
    return reinterpret_cast<vital_bounding_box_t*>(
      new kwiver::vital::bounding_box_d( ulp, lrp ) );
  );
  return 0;
}

// ------------------------------------------------------------------
vital_bounding_box_t* vital_bounding_box_new_from_point_width_height( double* ul,
                                                                      double  width,
                                                                      double height)
{
  const point_t ulp( ul[0], ul[1] );
  STANDARD_CATCH(
    "vital_bounding_box_new_from_point_width_height", 0,
    return reinterpret_cast<vital_bounding_box_t*>(
      new kwiver::vital::bounding_box_d( ulp, width, height ) );
  );
  return 0;
}

// ------------------------------------------------------------------
vital_bounding_box_t* vital_bounding_box_new_from_coordinates( double xmin,
                                                               double ymin,
                                                               double xmax,
                                                               double ymax )
{
  STANDARD_CATCH(
    "vital_bounding_box_new_from_coordinates", 0,
    return reinterpret_cast<vital_bounding_box_t*>(
      new kwiver::vital::bounding_box_d( xmin, ymin, xmax, ymax ) );
  );
  return 0;
}

// ------------------------------------------------------------------
vital_bounding_box_t* vital_bounding_box_copy( vital_bounding_box_t* bbox )
{
  STANDARD_CATCH(
    "vital_bounding_box_copy", 0,
    vital_bounding_box_t* item = reinterpret_cast<vital_bounding_box_t*>(
      new kwiver::vital::bounding_box_d(
        *reinterpret_cast<kwiver::vital::bounding_box_d*>(bbox) ) );
    return item;
  );
  return 0;
}

// ------------------------------------------------------------------
void vital_bounding_box_destroy( vital_bounding_box_t* bbox )
{
  STANDARD_CATCH(
    "vital_bounding_box_destroy", 0,
    REINTERP_TYPE( kwiver::vital::bounding_box_d, bbox, b_ptr );
    delete b_ptr;
  );
};

// ------------------------------------------------------------------
//
// A little shortcut for defining accessors
//
#define ACCESSOR( TYPE, NAME )                                          \
TYPE vital_bounding_box_ ## NAME( vital_bounding_box_t *bbox )          \
{                                                                       \
  STANDARD_CATCH(                                                       \
    "vital_bounding_box_" # NAME, 0,                                    \
    return reinterpret_cast<kwiver::vital::bounding_box_d*>(bbox)->NAME(); \
  );                                                                    \
  return 0;                                                             \
}

double* vital_bounding_box_center( vital_bounding_box_t* bbox)
{
  STANDARD_CATCH(
    "vital_bounding_box_center", 0,
    point_t ctr = reinterpret_cast<kwiver::vital::bounding_box_d*>(bbox)->center();
    double* retval = new double[2];
    retval[0] = ctr(0);
    retval[1] = ctr(1);
    return retval;
  );
  return 0;
}

double* vital_bounding_box_upper_left( vital_bounding_box_t* bbox)
{
  STANDARD_CATCH(
    "vital_bounding_box_upper_left", 0,
    point_t ctr = reinterpret_cast<kwiver::vital::bounding_box_d*>(bbox)->upper_left();
    double* retval = new double[2];
    retval[0] = ctr(0);
    retval[1] = ctr(1);
    return retval;
  );
  return 0;
}

double* vital_bounding_box_lower_right( vital_bounding_box_t* bbox)
{
  STANDARD_CATCH(
    "vital_bounding_box_lower_right", 0,
    point_t ctr = reinterpret_cast<kwiver::vital::bounding_box_d*>(bbox)->lower_right();
    double* retval = new double[2];
    retval[0] = ctr[0];
    retval[1] = ctr[1];
    return retval;
  );
  return 0;
}

ACCESSOR( double, min_x )
ACCESSOR( double, max_x )
ACCESSOR( double, min_y )
ACCESSOR( double, max_y )
ACCESSOR( double, width )
ACCESSOR( double, height )
ACCESSOR( double, area )

#undef ACCESSOR
