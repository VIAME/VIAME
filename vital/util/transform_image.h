// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief templated image transformation functions
 */

#ifndef VITAL_TRANSFORM_IMAGE_H_
#define VITAL_TRANSFORM_IMAGE_H_

#include <vital/types/image.h>
#include <cstdlib>

namespace kwiver {
namespace vital {

/// Transform a given image in place given a unary function
/**
 * Apply a given unary function to all pixels in the image. This is guarateed
 * to traverse the pixels in an optimal order, i.e. in-memory-order traversal.
 *
 * Example:
\code
static kwiver::vital::image::byte invert_mask_pixel( kwiver::vital::image::byte const &b )
{ return !b; }

kwiver::vital::image   mask_img( mask->get_image() );
kwiver::vital::transform_image( mask_img, invert_mask_pixel );

// or as a functor
class multiply_by {
private:
    int factor;

public:
    multiply_by(int x) : factor(x) { }

    kwiver::vital::image::byte   operator () (kwiver::vital::image::byte const& other) const
    {
        return factor * other;
    }
};

kwiver::vital::transform_image( mask_img, multiply_by( 5 ) );

\endcode
 *
 * \param img Input image reference to transform the data of
 * \param op Unary function which takes a const byte& and returns a byte
 */
template <typename T, typename OP>
void transform_image( image_of<T>& img, OP op )
{
  // determine which order to traverse dimensions
  // [0] -> smalled distance between values
  // [2] -> greatest distance between values
  size_t side_len[3];
  ptrdiff_t step_size[3];
  bool wBh = std::abs(img.w_step()) < std::abs(img.h_step()),
       dBh = std::abs(img.d_step()) < std::abs(img.h_step()),
       dBw = std::abs(img.d_step()) < std::abs(img.w_step());
  size_t w_idx = static_cast<size_t>( ! wBh ) + static_cast<size_t>(dBw),
         h_idx = static_cast<size_t>(wBh) + static_cast<size_t>(dBh),
         d_idx = static_cast<size_t>( ! dBw ) + static_cast<size_t>( ! dBh );

  side_len[w_idx] = img.width();
  side_len[h_idx] = img.height();
  side_len[d_idx] = img.depth();
  step_size[w_idx] = img.w_step();
  step_size[h_idx] = img.h_step();
  step_size[d_idx] = img.d_step();

  // position index with a dimension
  unsigned i0, i1, i2;
  // Pointers to the first pixel of the current dimension iteration
  T* d0_s, * d1_s, * d2_s;

  d2_s = img.first_pixel();
  for ( i2 = 0; i2 < side_len[2]; ++i2, d2_s += step_size[2] )
  {
    d1_s = d2_s;
    for ( i1 = 0; i1 < side_len[1]; ++i1, d1_s += step_size[1] )
    {
      d0_s = d1_s;
      for ( i0 = 0; i0 < side_len[0]; ++i0, d0_s += step_size[0] )
      {
        *d0_s = op( *d0_s );
      }
    }
  }
}

/// Transform an input image to an output image given a unary function
/**
 * This function is similar to the inplace variant except it copies the transformed data
 * from one const image to another.  The input and ouput images must have the same dimensions
 * but can have different types and memory layouts.  If the output image does not have the
 * correct dimensions its memory will be reallocated to match the dimensions of the input image.
 */
template <typename T1, typename T2, typename OP>
void transform_image( image_of<T1> const& img_in, image_of<T2>& img_out, OP op )
{
  const unsigned width = static_cast<unsigned int>(img_in.width());
  const unsigned height = static_cast<unsigned int>(img_in.height());
  const unsigned depth = static_cast<unsigned int>(img_in.depth());

  // make sure the output image has the same size as the input image
  img_out.set_size(width, height, depth);
  // Pointers to the first pixel of the current dimension iteration
  T1 const* d0_i, * d1_i, * d2_i;
  T2* d0_o, * d1_o, * d2_o;

  d2_i = img_in.first_pixel();
  d2_o = img_out.first_pixel();
  for ( unsigned i2 = 0; i2 < depth; ++i2, d2_i += img_in.d_step(), d2_o += img_out.d_step() )
  {
    d1_i = d2_i;
    d1_o = d2_o;
    for ( unsigned i1 = 0; i1 < height; ++i1, d1_i += img_in.h_step(), d1_o += img_out.h_step() )
    {
      d0_i = d1_i;
      d0_o = d1_o;
      for ( unsigned i0 = 0; i0 < width; ++i0, d0_i += img_in.w_step(), d0_o += img_out.w_step() )
      {
        *d0_o = op( *d0_i );
      }
    }
  }
}

/// Functor for casting pixel values for use in the cast_image function
template <typename T1, typename T2>
struct cast_pixel
{
  T2 operator () (T1 const& v) const { return static_cast<T2>(v); }
};

/// Specialization of cast_pixel for bool to avoid compiler warnings
template <typename T1>
struct cast_pixel<T1, bool>
{
  bool operator () (T1 const& v) const { return v != T1(0); }
};

/// Static cast an image of one type to that of another type
template <typename T1, typename T2>
void cast_image( image_of<T1> const& img_in, image_of<T2>& img_out )
{
  transform_image(img_in, img_out, cast_pixel<T1,T2>());
}

/// Static cast an image of unknown type to a known type
template <typename T>
void cast_image( image const& img_in, image_of<T>& img_out )
{
  if( img_in.pixel_traits() == image_pixel_traits_of<T>() )
  {
    // if the types are already the same then shallow copy
    img_out = img_in;
    return;
  }
#define TRY_TYPE(in_T)                                         \
  if( img_in.pixel_traits() == image_pixel_traits_of<in_T>() ) \
  {                                                            \
    cast_image( image_of<in_T>(img_in), img_out );             \
    return;                                                    \
  }

  TRY_TYPE(bool)
  TRY_TYPE(uint8_t)
  TRY_TYPE(int8_t)
  TRY_TYPE(uint16_t)
  TRY_TYPE(int16_t)
  TRY_TYPE(uint32_t)
  TRY_TYPE(int32_t)
  TRY_TYPE(uint64_t)
  TRY_TYPE(int64_t)
  TRY_TYPE(float)
  TRY_TYPE(double)

#undef TRY_TYPE

  VITAL_THROW( image_type_mismatch_exception,
               "kwiver::vital::cast_image() cannot cast unknown type");
}

/// Call a unary function on every pixel in a const image
/**
 * Apply a given unary function to all pixels in the image. This is guarateed
 * to traverse the pixels in an optimal order, i.e. in-memory-order traversal.
 *
 * Example:
\code
kwiver::vital::image_of<uint_8>my_image( img->get_image() );
uint8_t max_v = 0;
// using a lambda function to get the maximum pixel value
kwiver::vital::foreach_pixel( my_image, [&max_v](uint8_t p)
{
  max_v = std::max(max_v, p)
});

\endcode
 *
 * \param img Input image reference
 * \param op Unary function which takes the pixel type
 */
template <typename T, typename OP>
void foreach_pixel( image_of<T> const& img, OP op )
{
  // determine which order to traverse dimensions
  // [0] -> smalled distance between values
  // [2] -> greatest distance between values
  size_t side_len[3];
  ptrdiff_t step_size[3];
  bool wBh = std::abs(img.w_step()) < std::abs(img.h_step()),
       dBh = std::abs(img.d_step()) < std::abs(img.h_step()),
       dBw = std::abs(img.d_step()) < std::abs(img.w_step());
  size_t w_idx = static_cast<size_t>( ! wBh ) + static_cast<size_t>(dBw),
         h_idx = static_cast<size_t>(wBh) + static_cast<size_t>(dBh),
         d_idx = static_cast<size_t>( ! dBw ) + static_cast<size_t>( ! dBh );

  side_len[w_idx] = img.width();
  side_len[h_idx] = img.height();
  side_len[d_idx] = img.depth();
  step_size[w_idx] = img.w_step();
  step_size[h_idx] = img.h_step();
  step_size[d_idx] = img.d_step();

  // position index with a dimension
  unsigned i0, i1, i2;
  // Pointers to the first pixel of the current dimension iteration
  T const* d0_s, * d1_s, * d2_s;

  d2_s = img.first_pixel();
  for ( i2 = 0; i2 < side_len[2]; ++i2, d2_s += step_size[2] )
  {
    d1_s = d2_s;
    for ( i1 = 0; i1 < side_len[1]; ++i1, d1_s += step_size[1] )
    {
      d0_s = d1_s;
      for ( i0 = 0; i0 < side_len[0]; ++i0, d0_s += step_size[0] )
      {
        op( *d0_s );
      }
    }
  }
}

} }   // end namespace vital

#endif // VITAL_TRANSFORM_IMAGE_H_
