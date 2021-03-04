// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of C interface to vital::rgb_color
 */

#include "color.h"

#include <vital/types/color.h>

#include <vital/bindings/c/helpers/c_utils.h>

using namespace kwiver;

/// Create a new rgb_color instance
vital_rgb_color_t*
vital_rgb_color_new( unsigned char cr, unsigned char cg, unsigned char cb,
                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_new", eh,
    return reinterpret_cast< vital_rgb_color_t* >(
      new vital::rgb_color(cr, cg, cb)
    );
  );
  return 0;
}

/// Create a new default (white) rgb_color instance
vital_rgb_color_t*
vital_rgb_color_new_default( vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_new_default", eh,
    return reinterpret_cast< vital_rgb_color_t* >(
      new vital::rgb_color()
    );
  );
  return 0;
}

/// Destroy an rgb_color instance
void
vital_rgb_color_destroy( vital_rgb_color_t *c, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_destroy", eh,
    REINTERP_TYPE( vital::rgb_color, c, c_ptr );
    delete c_ptr;
  );
}

/// Get the red value
unsigned char
vital_rgb_color_r( vital_rgb_color_t *c, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_r", eh,
    REINTERP_TYPE( vital::rgb_color, c, cptr );
    return cptr->r;
  );
  return 0;
}

/// Get the green value
unsigned char
vital_rgb_color_g( vital_rgb_color_t *c, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_g", eh,
    REINTERP_TYPE( vital::rgb_color, c, cptr );
    return cptr->g;
  );
  return 0;
}

/// Get the blue value
unsigned char
vital_rgb_color_b( vital_rgb_color_t *c, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_b", eh,
    REINTERP_TYPE( vital::rgb_color, c, cptr );
    return cptr->b;
  );
  return 0;
}

/// Test equality between two rgb_color instances
bool
vital_rgb_color_is_equal( vital_rgb_color_t *a, vital_rgb_color_t *b,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_rgb_color_is_equal", eh,
    REINTERP_TYPE( vital::rgb_color, a, aptr );
    REINTERP_TYPE( vital::rgb_color, b, bptr );
    return ((*aptr) == (*bptr));
  );
  return false;
}
