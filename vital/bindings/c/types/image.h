// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::image classes
 */

#ifndef VITAL_C_IMAGE_H_
#define VITAL_C_IMAGE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <vital/bindings/c/vital_c_export.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef unsigned char vital_image_byte;

/// VITAL Image opaque structure
typedef struct vital_image_s vital_image_t;

/// Enum for different pixel types in an image
enum vital_image_pixel_type_t {VITAL_PIXEL_UNKNOWN = 0,
                               VITAL_PIXEL_UNSIGNED = 1,
                               VITAL_PIXEL_SIGNED = 2,
                               VITAL_PIXEL_FLOAT = 3,
                               VITAL_PIXEL_BOOL = 4};

/// Create a new, empty image
VITAL_C_EXPORT
vital_image_t* vital_image_new();

/// Create a new image with dimensions and type, allocating memory
/**
 * @param width Width of image in pixels
 * @param height Height of image in pixels
 * @param depth Number of planes in image
 * @param interleave If true change the memory layout to interleave the channels (depth)
 * @param pixel_type The class of data type used for the pixels
 * @param pixel_num_bytes The number of bytes use to reperesent a pixel
 *
 * @return Opaque pointer to new image
 */
VITAL_C_EXPORT
vital_image_t* vital_image_new_with_dim( size_t width, size_t height,
                                         size_t depth, bool interleave,
                                         vital_image_pixel_type_t pixel_type,
                                         size_t pixel_num_bytes);

/// Create a new image wrapping existing data
/**
 * This function creates an image object from raw memory owned by the
 * caller.  The constructed image does not take ownership of the memory.
 *
 * @param first_pixel Address of first pixel (0, 0, 0)
 * @param width Width of image in pixels
 * @param height Height of image in pixels
 * @param depth Number of planes in image
 * @param w_step Increment to get to next column pixel (x)
 * @param h_step Increment to get to next row pixel (y)
 * @param d_step Increment to get to pixel in next plane
 * @param pixel_type The class of data type used for the pixels
 * @param pixel_num_bytes The number of bytes use to reperesent a pixel
 *
 * @return Opaque pointer to new image
 */
VITAL_C_EXPORT
vital_image_t* vital_image_new_from_data( void const* first_pixel,
                                          size_t width, size_t height, size_t depth,
                                          int32_t w_step, int32_t h_step, int32_t d_step,
                                          vital_image_pixel_type_t pixel_type,
                                          size_t pixel_num_bytes);

/// Create a new image from an existing image, sharing the same memory
/**
 * The new image will share the same memory as the old image
 */
VITAL_C_EXPORT
vital_image_t* vital_image_new_from_image( vital_image_t *other_image );

/// Destroy an image instance
VITAL_C_EXPORT
void vital_image_destroy( vital_image_t* image );

/// Copy the data from \p image_src into \p image_dest
/**
 * The if destination image does not match the source image in size
 * or pixel type, the destination image will be reallocated to match
 *
 * @param image_dest The destination of the image copy
 * @param image_src  The source image to copy
 */
VITAL_C_EXPORT
void vital_image_copy_from_image(vital_image_t *image_dest,
                                 vital_image_t *image_src );

/// Get the number of bytes allocated in the given image
VITAL_C_EXPORT
size_t vital_image_size( vital_image_t* image );

/// Get first pixel address
VITAL_C_EXPORT
void* vital_image_first_pixel( vital_image_t* image );

/// Get image width
VITAL_C_EXPORT
size_t vital_image_width( vital_image_t* image );

/// Get image height
VITAL_C_EXPORT
size_t vital_image_height( vital_image_t* image );

/// Get image depth
VITAL_C_EXPORT
size_t vital_image_depth( vital_image_t* image );

/// Get number of bytes in a pixel
VITAL_C_EXPORT
size_t vital_image_pixel_num_bytes( vital_image_t* image );

/// Return the type enum of a pixel
VITAL_C_EXPORT
vital_image_pixel_type_t vital_image_pixel_type( vital_image_t* image );

/// Get image w_step
VITAL_C_EXPORT
size_t vital_image_w_step( vital_image_t* image );

/// Get image h_step
VITAL_C_EXPORT
size_t vital_image_h_step( vital_image_t* image );

/// Get image d_step
VITAL_C_EXPORT
size_t vital_image_d_step( vital_image_t* image );

/// Get image d_step
VITAL_C_EXPORT
bool vital_image_is_contiguous( vital_image_t* image );

/// Return true if two images have equal content (deep equality)
VITAL_C_EXPORT
bool vital_image_equal_content( vital_image_t* image1, vital_image_t* image2 );

/// Get pixel value at location (i,j) assuming a single channel unsigned 8-bit image
VITAL_C_EXPORT
uint8_t vital_image_get_pixel2_uint8( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an unsigned 8-bit image
VITAL_C_EXPORT
uint8_t vital_image_get_pixel3_uint8( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel signed 8-bit image
VITAL_C_EXPORT
int8_t vital_image_get_pixel2_int8( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an signed 8-bit image
VITAL_C_EXPORT
int8_t vital_image_get_pixel3_int8( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel unsigned 16-bit image
VITAL_C_EXPORT
uint16_t vital_image_get_pixel2_uint16( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an unsigned 16-bit image
VITAL_C_EXPORT
uint16_t vital_image_get_pixel3_uint16( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel signed 16-bit image
VITAL_C_EXPORT
int16_t vital_image_get_pixel2_int16( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an signed 16-bit image
VITAL_C_EXPORT
int16_t vital_image_get_pixel3_int16( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel unsigned 32-bit image
VITAL_C_EXPORT
uint32_t vital_image_get_pixel2_uint32( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an unsigned 32-bit image
VITAL_C_EXPORT
uint32_t vital_image_get_pixel3_uint32( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel signed 32-bit image
VITAL_C_EXPORT
int32_t vital_image_get_pixel2_int32( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an signed 32-bit image
VITAL_C_EXPORT
int32_t vital_image_get_pixel3_int32( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel unsigned 64-bit image
VITAL_C_EXPORT
uint64_t vital_image_get_pixel2_uint64( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an unsigned 64-bit image
VITAL_C_EXPORT
uint64_t vital_image_get_pixel3_uint64( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel signed 64-bit image
VITAL_C_EXPORT
int64_t vital_image_get_pixel2_int64( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming an signed 64-bit image
VITAL_C_EXPORT
int64_t vital_image_get_pixel3_int64( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel float image
VITAL_C_EXPORT
float vital_image_get_pixel2_float( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming a float image
VITAL_C_EXPORT
float vital_image_get_pixel3_float( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel double image
VITAL_C_EXPORT
double vital_image_get_pixel2_double( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming a double image
VITAL_C_EXPORT
double vital_image_get_pixel3_double( vital_image_t *image, unsigned i, unsigned j, unsigned k );

/// Get pixel value at location (i,j) assuming a single channel bool image
VITAL_C_EXPORT
bool vital_image_get_pixel2_bool( vital_image_t *image, unsigned i, unsigned j );

/// Get pixel value at location (i,j,k) assuming a bool image
VITAL_C_EXPORT
bool vital_image_get_pixel3_bool( vital_image_t *image, unsigned i, unsigned j, unsigned k );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_IMAGE_H_
