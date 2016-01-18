/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief vital::image C interface implementation
 */

#include "image.h"

#include <vital/types/image.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <iostream>


/// Create a new, empty image
vital_image_t* vital_image_new()
{
  STANDARD_CATCH(
    "C::image:new", 0,
    return reinterpret_cast<vital_image_t*>( new kwiver::vital::image() );
  );
  return 0;
}


/// Create a new image with dimensions, allocating memory
vital_image_t* vital_image_new_with_dim( size_t width, size_t height,
                                         size_t depth, bool interleave )
{
  STANDARD_CATCH(
    "C::image:new_with_dim", 0,
    return reinterpret_cast<vital_image_t*>(
      new kwiver::vital::image( width, height, depth, interleave )
      );
  );
  return 0;
}


/// Create a new image from new data
vital_image_t* vital_image_new_from_data( unsigned char const *first_pixel,
                                          size_t width, size_t height, size_t depth,
                                          int32_t w_step, int32_t h_step, int32_t d_step )
{
  kwiver::vital::image* new_image = new kwiver::vital::image( width, height, depth );
  for ( unsigned int d = 0; d < depth; ++d )
  {
    // This is a little crazy, but it is used to convert BGR to RGB
    int d_idx(0);
    if (d_step >= 0)
    {
      d_idx = d * d_step;
    }
    else
    {
      d_idx = static_cast< int >( (depth - d - 1) * (-d_step) );
    }

    for ( unsigned int h = 0; h < height; ++h )
    {
      int h_idx = h * h_step;
      for ( unsigned int w = 0; w < width; ++w )
      {
        int w_idx = w * w_step;
        (*new_image)( w, h, d ) = first_pixel[ w_idx + h_idx + d_idx ];
      }
    }
  }
  return reinterpret_cast<vital_image_t*>( new_image );
}


/// Create a new image from an existing image
vital_image_t* vital_image_new_from_image( vital_image_t *other_image )
{
  STANDARD_CATCH(
    "C::image::new_from_image", 0,
    return reinterpret_cast<vital_image_t*>(
      new kwiver::vital::image( *reinterpret_cast<kwiver::vital::image*>(other_image) )
      );
  );
  return 0;
}


/// Destroy an image instance
void vital_image_destroy( vital_image_t *image )
{
  STANDARD_CATCH(
    "C::image::destroy", 0,
    delete reinterpret_cast<kwiver::vital::image*>( image );
  );
};


int vital_image_get_pixel2( vital_image_t *image, unsigned i, unsigned j )
{
  STANDARD_CATCH(
    "C::image::destroy", 0,
    return reinterpret_cast<kwiver::vital::image*>( image )->operator()(i, j);
  );
  return 0;
}


int vital_image_get_pixel3( vital_image_t *image, unsigned i, unsigned j, unsigned k )
{
  STANDARD_CATCH(
    "C::image::destroy", 0,
    return reinterpret_cast<kwiver::vital::image*>( image )->operator()(i, j, k);
  );
  return 0;
}

//
// A little shortcut for defining accessors
//
#define ACCESSOR( TYPE, NAME )                                          \
TYPE vital_image_ ## NAME( vital_image_t *image )                       \
{                                                                       \
  STANDARD_CATCH(                                                       \
    "C::image::" # NAME, 0,                                             \
    return reinterpret_cast<kwiver::vital::image*>(image)->NAME();      \
  );                                                                    \
  return 0;                                                             \
}

/// Get the number of bytes allocated in the given image
ACCESSOR( size_t, size )
ACCESSOR( vital_image_byte*, first_pixel )
ACCESSOR( size_t, width )
ACCESSOR( size_t, height )
ACCESSOR( size_t, depth )
ACCESSOR( size_t, w_step )
ACCESSOR( size_t, h_step )
ACCESSOR( size_t, d_step )

#undef ACCESSOR
