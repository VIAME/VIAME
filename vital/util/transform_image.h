/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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
 * Apply a given unary function to all pixels in the image. This is guareteed
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
VITAL_EXPORT void transform_image( image_of<T>& img, OP op )
{
  // determine which order to traverse dimensions
  // [0] -> smalled distance between values
  // [2] -> greatest distance between values
  size_t side_len[3];
  ptrdiff_t step_size[3];
  bool wBh = std::abs(img.w_step()) < std::abs(img.h_step()),
       dBh = std::abs(img.d_step()) < std::abs(img.h_step()),
       dBw = std::abs(img.d_step()) < std::abs(img.w_step());
  size_t w_idx = ( ! wBh ) + dBw,
         h_idx = wBh + dBh,
         d_idx = ( ! dBw ) + ( ! dBh );

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

} }   // end namespace vital


#endif // VITAL_TRANSFORM_IMAGE_H_
