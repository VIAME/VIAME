/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief core image class interface
 */

#ifndef VITAL_IMAGE_H_
#define VITAL_IMAGE_H_

#include <vital/types/color.h>

#include <vital/vital_export.h>

#include <memory>
#include <stdexcept>

#include <cstddef>

namespace kwiver {
namespace vital {

/// Basic in memory image.
/**
 * This class represents an image with byte wide pixels in a block of
 * image memory on the heap.
 *
 * The image object uses shared pointers to this class. The image
 * memory is separated from the image object so it can be shared among
 * many image objects.
 *
 * Derived image memory classes can proved access to image memory
 * stored in other forms, such as on the GPU or in 3rd party data structures.
 */
class VITAL_EXPORT image_memory
{
public:
  /// Default Constructor
  image_memory();

  /// Constructor - allocates n bytes
  /**
   * \param n bytes to allocate
   */
  image_memory( size_t n );

  /// Copy constructor
  /**
   * \param other The other image_memory to copy from.
   */
  image_memory( const image_memory& other );

  /// Assignment operator
  /**
   * Other image_memory whose internal data is copied into ours.
   * \param other image_memory to copy from.
   */
  image_memory& operator=( const image_memory& other );

  /// Destructor
  virtual ~image_memory();

  /// Return a pointer to the allocated memory
  virtual void* data();

  /// The number of bytes allocated
  size_t size() const { return size_; }


protected:
  /// The image data
  void* data_;

  /// The number of bytes allocated
  size_t size_;
};

/// Shared pointer for base image_memory type
typedef std::shared_ptr< image_memory > image_memory_sptr;


// ==================================================================
/// The representation of an in-memory image.
/**
 * Images share memory using the image_memory class.  This is
 * effectively a view on an image.
 */
class VITAL_EXPORT image
{
public:
  /// Convenience typedef for the size of a byte
  typedef unsigned char byte;

  /// Default Constructor
  image();

  /// Constructor that allocates image memory
  /**
   * Create a new blank (empty) image of specified size.
   *
   * \param width Number of pixels in width
   * \param height Number of pixel rows
   * \param depth Number of image channels
   * \param interleave Set if the pixels are interleaved
   */
  image( size_t width, size_t height, size_t depth = 1, bool interleave = false );

  /// Constructor that points at existing memory
  /**
   * Create a new image from supplied memory.
   *
   * \param first_pixel Address of the first pixel in the image. This
   * does not have to be the lowest memory address of the image
   * memory.
   *
   * \param width Number of pixels wide
   * \param height Number of pixels high
   * \param depth Number of image channels
   * \param w_step Byte increment to get to next pixel column
   * \param h_step Byte increment to get to next pixel row
   * \param d_step Byte increment to get to next image channel
   */
  image( const byte* first_pixel, size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step );

  /// Constructor that shares memory with another image
  /**
   * Create a new image from existing image.
   *
   * \param mem Address of the first pixel in the image. This
   * does not have to be the lowest memory address of the image
   * memory.
   *
   * \param width Number of pixels wide
   * \param height Number of pixels high
   * \param depth Number of image channels
   * \param w_step Byte increment to get to next pixel column
   * \param h_step Byte increment to get to next pixel row
   * \param d_step Byte increment to get to next image channel
   */
  image( const image_memory_sptr& mem,
         const byte* first_pixel, size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step );

  /// Copy Constructor
  /**
   * The new image will share the same memory as the old image
   * \param other The other image.
   */
  image( const image& other );

  /// Assignment operator
  const image& operator=( const image& other );

  /// Const access to the image memory
  const image_memory_sptr& memory() const { return data_; }

  /// Access to the image memory
  image_memory_sptr memory() { return data_; }

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  size_t size() const;

  /// Const access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a large image memory chunk.
   */
  const byte* first_pixel() const { return first_pixel_; }

  /// Access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a larger image memory chunk.
   */
  byte* first_pixel() { return first_pixel_; }

  /// The width of the image in pixels
  size_t width() const { return width_; }

  /// The height of the image in pixels
  size_t height() const { return height_; }

  /// The depth (or number of channels) of the image
  size_t depth() const { return depth_; }

  /// The the step in memory to next pixel in the width direction
  ptrdiff_t w_step() const { return w_step_; }

  /// The the step in memory to next pixel in the height direction
  ptrdiff_t h_step() const { return h_step_; }

  /// The the step in memory to next pixel in the depth direction
  ptrdiff_t d_step() const { return d_step_; }

  /// Const access pixels in the image
  /**
   * This returns the specified pixel in the image as an rgb_color. This
   * assumes that the image is either Luminance[, Alpha], if depth() < 3, and
   * that only the first (Luminance) channel is interesting (in which case the
   * R, G, B values of the returned rgb_color are all taken from the first
   * channel), or that only the first three channels are interesting, and are
   * in the order Red, Green, Blue.
   *
   * \param i width position (x)
   * \param j height position (y)
   */
  inline rgb_color at( unsigned i, unsigned j ) const
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at(unsigned, unsigned) const");
    }

    if ( depth_ < 3 )
    {
      auto const v = first_pixel_[w_step_ * i + h_step_ * j];
      return { v, v, v };
    }

    auto const r = first_pixel_[w_step_ * i + h_step_ * j + d_step_ * 0];
    auto const g = first_pixel_[w_step_ * i + h_step_ * j + d_step_ * 1];
    auto const b = first_pixel_[w_step_ * i + h_step_ * j + d_step_ * 2];
    return { r, g, b };
  }


  /// Access pixels in the first channel of the image
  /**
   * \param i width position (x)
   * \param j height position (y)
   */
  inline byte& operator()( unsigned i, unsigned j )
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::operator()(unsigned, unsigned)");
    }
    return first_pixel_[w_step_ * i + h_step_ * j];
  }


  /// Const access pixels in the first channel of the image
  inline const byte& operator()( unsigned i, unsigned j ) const
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::operator()(unsigned, unsigned) const");
    }
    return first_pixel_[w_step_ * i + h_step_ * j];
  }


  /// Access pixels in the image (width, height, channel)
  inline byte& operator()( unsigned i, unsigned j, unsigned k )
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::operator()(unsigned, unsigned, unsigned)");
    }
    return first_pixel_[w_step_ * i + h_step_ * j + d_step_ * k];
  }


  /// Const access pixels in the image (width, height, channel)
  inline const byte& operator()( unsigned i, unsigned j, unsigned k ) const
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::operator()(unsigned, unsigned, unsigned) const");
    }
    return first_pixel_[w_step_ * i + h_step_ * j + d_step_ * k];
  }


  /// Deep copy the image data from another image into this one
  void copy_from( const image& other );

  /// Set the size of the image.
  /**
   * If the size has not changed, do nothing.
   * Otherwise, allocate new memory matching the new size.
   * \param width a new image width
   * \param height a new image height
   * \param depth a new image depth
   */
  void set_size( size_t width, size_t height, size_t depth );


protected:
  /// Smart pointer to memory viewed by this class
  image_memory_sptr data_;
  /// Pointer to the pixel at the origin
  byte* first_pixel_;
  /// Width of the image
  size_t width_;
  /// Height of the image
  size_t height_;
  /// Depth of the image (i.e. number of channels)
  size_t depth_;
  /// Increment to move to the next pixel along the width direction
  ptrdiff_t w_step_;
  /// Increment to move to the next pixel along the height direction
  ptrdiff_t h_step_;
  /// Increment to move to the next pixel along the depth direction
  ptrdiff_t d_step_;

};


/// Compare to images to see if the pixels have the same values.
/**
 * This does not require that the images have the same memory layout,
 * only that the images have the same dimensions and pixel values.
 * \param img1 first image to compare
 * \param img2 second image to compare
 */
VITAL_EXPORT bool equal_content( const image& img1, const image& img2 );


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
VITAL_EXPORT void transform_image( image& img,
                                   image::byte ( * op )( image::byte const& ) );


} }   // end namespace vital


#endif // VITAL_IMAGE_H_
