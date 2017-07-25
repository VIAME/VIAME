/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
