// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::image C interface implementation
 */

#include "image.h"

#include <vital/vital_config.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/types/image.h>

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
                                         size_t depth, bool interleave,
                                         vital_image_pixel_type_t pixel_type,
                                         size_t pixel_num_bytes)
{
  STANDARD_CATCH(
    "C::image:new_with_dim", 0,
    typedef kwiver::vital::image_pixel_traits pixel_traits;
    pixel_traits pt(static_cast<pixel_traits::pixel_type>(pixel_type), pixel_num_bytes);
    return reinterpret_cast<vital_image_t*>(
      new kwiver::vital::image( width, height, depth,
                                interleave, pt )
      );
  );
  return 0;
}

/// Create a new image wrapping existing data
vital_image_t* vital_image_new_from_data( void const* first_pixel,
                                          size_t width, size_t height, size_t depth,
                                          int32_t w_step, int32_t h_step, int32_t d_step,
                                          vital_image_pixel_type_t pixel_type,
                                          size_t pixel_num_bytes)
{
  STANDARD_CATCH(
    "C::image:new_from_data", 0,
    typedef kwiver::vital::image_pixel_traits pixel_traits;
    pixel_traits pt(static_cast<pixel_traits::pixel_type>(pixel_type), pixel_num_bytes);
    return reinterpret_cast<vital_image_t*>(
      new kwiver::vital::image( first_pixel, width, height, depth,
                                w_step, h_step, d_step, pt )
      );
  );
  return 0;
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

/// Copy the data from \p image_src into \p image_dest
void vital_image_copy_from_image(vital_image_t *image_dest,
                                 vital_image_t *image_src )
{
  STANDARD_CATCH(
    "C::image::copy_from", 0,
    reinterpret_cast<kwiver::vital::image*>( image_dest )->copy_from(
      *reinterpret_cast<kwiver::vital::image*>( image_src ));
  );
}

/// Return true if two images have equal content (deep equality)
VITAL_C_EXPORT
bool vital_image_equal_content( vital_image_t* image1, vital_image_t* image2 )
{
  STANDARD_CATCH(
    "C::image::equal_content", 0,
    return kwiver::vital::equal_content(*reinterpret_cast<kwiver::vital::image*>( image1 ),
                                        *reinterpret_cast<kwiver::vital::image*>( image2 ));
  );
  return false;
}

//
// A little shortcut for defining pixel accessors
//
#define GET_PIXEL( TYPE, NAME )                                             \
TYPE vital_image_get_pixel2_ ## NAME( vital_image_t *image,                 \
                                      unsigned i, unsigned j )              \
{                                                                           \
  STANDARD_CATCH(                                                           \
    "C::image::get_pixel2_" # NAME, 0,                                      \
    return reinterpret_cast<kwiver::vital::image*>(image)->at<TYPE>(i,j);   \
  );                                                                        \
  return static_cast<TYPE>(0);                                              \
}                                                                           \
TYPE vital_image_get_pixel3_ ## NAME( vital_image_t *image,                 \
                                      unsigned i, unsigned j,               \
                                      VITAL_UNUSED unsigned k )             \
{                                                                           \
  STANDARD_CATCH(                                                           \
    "C::image::get_pixel2_" # NAME, 0,                                      \
    return reinterpret_cast<kwiver::vital::image*>(image)->at<TYPE>(i,j);   \
  );                                                                        \
  return static_cast<TYPE>(0);                                              \
}

GET_PIXEL( uint8_t,  uint8 )
GET_PIXEL( int8_t,   int8 )
GET_PIXEL( uint16_t, uint16 )
GET_PIXEL( int16_t,  int16 )
GET_PIXEL( uint32_t, uint32 )
GET_PIXEL( int32_t,  int32 )
GET_PIXEL( uint64_t, uint64 )
GET_PIXEL( int64_t,  int64 )
GET_PIXEL( float,    float )
GET_PIXEL( double,   double )
GET_PIXEL( bool,     bool )

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

ACCESSOR( size_t, size )
ACCESSOR( void*, first_pixel )
ACCESSOR( size_t, width )
ACCESSOR( size_t, height )
ACCESSOR( size_t, depth )
ACCESSOR( size_t, w_step )
ACCESSOR( size_t, h_step )
ACCESSOR( size_t, d_step )
ACCESSOR( bool, is_contiguous )

#undef ACCESSOR

//
// A little shortcut for defining pixel traits accessors
//
#define PT_ACCESSOR( TYPE, NAME )                                       \
TYPE vital_image_pixel_ ## NAME( vital_image_t *image )                 \
{                                                                       \
  STANDARD_CATCH(                                                       \
    "C::image::pixel_format::" # NAME, 0,                               \
    return static_cast<TYPE>(                                           \
        reinterpret_cast<kwiver::vital::image*>(image)                  \
             ->pixel_traits().NAME);                                    \
  );                                                                    \
  return static_cast<TYPE>(0);                                          \
}

PT_ACCESSOR( size_t, num_bytes )
PT_ACCESSOR( vital_image_pixel_type_t, type )

#undef PT_ACCESSOR
