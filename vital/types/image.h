/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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
#include <vital/vital_types.h>
#include <vital/exceptions/image.h>

#include <memory>
#include <limits>
#include <stdexcept>

#include <cstddef>

namespace kwiver {
namespace vital {


/// A struct containing traits of the data type stored at each pixel
struct VITAL_EXPORT image_pixel_traits
{
  /// enumeration of the different type of pixel data
  enum pixel_type {UNKNOWN = 0, UNSIGNED = 1, SIGNED = 2, FLOAT = 3, BOOL = 4};

  /// Constructor - defaults to unsigned char (uint8) traits
  explicit image_pixel_traits( pixel_type t=UNSIGNED, size_t num_b=1 )
  : type(t), num_bytes(num_b) {}

  /// Equality operator
  bool operator==( const image_pixel_traits& other ) const
  {
    return this->type  == other.type &&
           this->num_bytes  == other.num_bytes;
  }

  /// Inequality operator
  bool operator!=( const image_pixel_traits& other ) const { return !(*this == other); }

  /// how we interpret this pixel
  pixel_type type;

  /// the number of bytes need to represent pixel data
  size_t num_bytes;
};

/// Output stream operator for image_pixel_traits::pixel_type
VITAL_EXPORT std::ostream& operator<<(std::ostream& os, image_pixel_traits::pixel_type pt);

/// Output stream operator for image_pixel_traits
VITAL_EXPORT std::ostream& operator<<(std::ostream& os, image_pixel_traits const& pt);


/// Helper struct to determine pixel type at compile time using std::numeric_limits
/*
 * This struct is an implementation detail and should generally not be use directly
 * \tparam V indicates that the type has a valid numeric_limits specialization
 * \tparam I indicates that this is an integer type
 * \tparam S indicates that this is a signed type
 */
template <bool V, bool I, bool S>
struct image_pixel_traits_helper;

/// Specialization of helper class for unknown types
template <bool I, bool S>
struct image_pixel_traits_helper<false, I, S>
{
  const static image_pixel_traits::pixel_type type = image_pixel_traits::UNKNOWN;
};

/// Specialization of helper class for signed types
template <>
struct image_pixel_traits_helper<true, true, true>
{
  const static image_pixel_traits::pixel_type type = image_pixel_traits::SIGNED;
};

/// Specialization of helper class for unsigned types
template <>
struct image_pixel_traits_helper<true, true, false>
{
  const static image_pixel_traits::pixel_type type = image_pixel_traits::UNSIGNED;
};

/// Specialization of helper class for floating point types
template <>
struct image_pixel_traits_helper<true, false, true>
{
  const static image_pixel_traits::pixel_type type = image_pixel_traits::FLOAT;
};


/// This class is used to instantiate an image_pixel_traits class based on type
/*
 * This class also contains \p static_type for compile-time look-up of the pixel_type
 * enum value by type.
 */
template <typename T>
struct image_pixel_traits_of : public image_pixel_traits
{
  VITAL_EXPORT
  const static image_pixel_traits::pixel_type static_type =
    image_pixel_traits_helper<std::numeric_limits<T>::is_specialized,
                              std::numeric_limits<T>::is_integer,
                              std::numeric_limits<T>::is_signed>::type;
  image_pixel_traits_of()
  : image_pixel_traits(static_type, sizeof(T)) {}
};

/// Specialization of image_pixel_traits_of for bool
template <>
struct image_pixel_traits_of<bool> : public image_pixel_traits
{
  VITAL_EXPORT
  const static image_pixel_traits::pixel_type static_type = image_pixel_traits::BOOL;
  image_pixel_traits_of<bool>()
  : image_pixel_traits(static_type, sizeof(bool)) {}
};


// ==================================================================
/// Provide compile-time look-up of data type from pixel_type enum and size
/*
 * This struct and its specializations provide compile-time mapping from
 * image_pixel_traits properties (pixel_type and num_bytes) to a concrete type
 */
template <image_pixel_traits::pixel_type T, size_t S>
struct image_pixel_from_traits
{ typedef void * type; };

#define image_pixel_from_traits_macro( T ) \
template <> struct VITAL_EXPORT \
image_pixel_from_traits<image_pixel_traits_of<T>::static_type, sizeof(T)> { typedef T type; }

image_pixel_from_traits_macro( uint8_t );
image_pixel_from_traits_macro( int8_t );
image_pixel_from_traits_macro( uint16_t );
image_pixel_from_traits_macro( int16_t );
image_pixel_from_traits_macro( uint32_t );
image_pixel_from_traits_macro( int32_t );
image_pixel_from_traits_macro( uint64_t );
image_pixel_from_traits_macro( int64_t );
image_pixel_from_traits_macro( float );
image_pixel_from_traits_macro( double );
image_pixel_from_traits_macro( bool );

#undef image_pixel_from_traits_macro


// ==================================================================
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
  
  /// Equality operator
  /**
   * Compares the data in other image memory with this image data.
   * \param other image_memory to compare with
   */
  bool operator==( const image_memory& other ) const;

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
  /// Default Constructor
  /**
   * \param pt Change the pixel traits of the image
   */
  image( const image_pixel_traits& pt=image_pixel_traits() );

  /// Constructor that allocates image memory
  /**
   * Create a new blank (empty) image of specified size.
   *
   * \param width Number of pixels in width
   * \param height Number of pixel rows
   * \param depth Number of image channels
   * \param pt data type traits of the image pixels
   * \param interleave Set if the pixels are interleaved
   */
  image( size_t width, size_t height, size_t depth = 1,
         bool interleave = false,
         const image_pixel_traits& pt=image_pixel_traits());

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
   * \param w_step pointer increment to get to next pixel column
   * \param h_step pointer increment to get to next pixel row
   * \param d_step pointer increment to get to next image channel
   * \param pt data type traits of the image pixels
   */
  image( const void* first_pixel,
         size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step,
         const image_pixel_traits& pt=image_pixel_traits() );

  /// Constructor that shares memory with another image
  /**
   * Create a new image from existing image.
   *
   * \param mem Shared memory block to be used
   * \param first_pixel Address of the first pixel in the image. This
   * does not have to be the lowest memory address of the image
   * memory.
   *
   * \param width Number of pixels wide
   * \param height Number of pixels high
   * \param depth Number of image channels
   * \param w_step pointer increment to get to next pixel column
   * \param h_step pointer increment to get to next pixel row
   * \param d_step pointer increment to get to next image channel
   * \param pt data type traits of the image pixels
   */
  image( const image_memory_sptr& mem,
         const void* first_pixel,
         size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step,
         const image_pixel_traits& pt=image_pixel_traits() );

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

  /// The size of the image managed data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth*bytes_per_pixel.
   *
   * \note This size only accounts for memory which is owned by
   *       the image.  If this image was constructed as a view
   *       into third party memory then the size is reported as 0.
   */
  size_t size() const;

  /// Const access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a large image memory chunk.
   */
  const void* first_pixel() const { return first_pixel_; }

  /// Access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a larger image memory chunk.
   */
  void* first_pixel() { return first_pixel_; }

  /// The width of the image in pixels
  size_t width() const { return width_; }

  /// The height of the image in pixels
  size_t height() const { return height_; }

  /// The depth (or number of channels) of the image
  size_t depth() const { return depth_; }

  /// The trait of the pixel data type
  const image_pixel_traits& pixel_traits() const { return pixel_traits_; }

  /// The the step in memory to next pixel in the width direction
  ptrdiff_t w_step() const { return w_step_; }

  /// The the step in memory to next pixel in the height direction
  ptrdiff_t h_step() const { return h_step_; }

  /// The the step in memory to next pixel in the depth direction
  ptrdiff_t d_step() const { return d_step_; }

  /// Return true if the pixels accessible in this image form a contiguous memory block
  bool is_contiguous() const;

  /// Equality operator
  /**
   * Compares this image to another image. Uses image data, pixel trait and image 
   * dimension for comparision
   * \param other image to compare with
   */
  bool operator==( const image& other_image ) const;

  /// Access pixels in the first channel of the image
  /**
   * \param i width position (x)
   * \param j height position (y)
   */
  template <typename T>
  inline T& at( unsigned i, unsigned j )
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(unsigned, unsigned)");
    }
    return reinterpret_cast<T*>(first_pixel_)[w_step_ * i + h_step_ * j];
  }


  /// Const access pixels in the first channel of the image
  template <typename T>
  inline const T& at( unsigned i, unsigned j ) const
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(unsigned, unsigned) const");
    }
    return reinterpret_cast<const T*>(first_pixel_)[w_step_ * i + h_step_ * j];
  }


  /// Access pixels in the image (width, height, channel)
  template <typename T>
  inline T& at( unsigned i, unsigned j, unsigned k )
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(unsigned, unsigned, unsigned)");
    }
    return reinterpret_cast<T*>(first_pixel_)[w_step_ * i + h_step_ * j + d_step_ * k];
  }


  /// Const access pixels in the image (width, height, channel)
  template <typename T>
  inline const T& at( unsigned i, unsigned j, unsigned k ) const
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(unsigned, unsigned, unsigned) const");
    }
    return reinterpret_cast<const T*>(first_pixel_)[w_step_ * i + h_step_ * j + d_step_ * k];
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
  void* first_pixel_;
  /// The traits of each pixel data type
  image_pixel_traits pixel_traits_;
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


// ==================================================================
/// The representation of an in-memory image.
/**
 * Images share memory using the image_memory class.  This is
 * effectively a view on an image.
 */
template <typename T>
class image_of : public image
{
public:
  /// Default Constructor
  image_of()
  : image(image_pixel_traits_of<T>()) {}

  // ----------------------------------------------------------------------------
  /// Constructor that allocates image memory
  /**
   * Create a new blank (empty) image of specified size.
   *
   * \param width Number of pixels in width
   * \param height Number of pixel rows
   * \param depth Number of image channels
   * \param interleave Set if the pixels are interleaved
   */
  image_of( size_t width, size_t height, size_t depth = 1, bool interleave = false )
  : image( width, height, depth, interleave, image_pixel_traits_of<T>() ) {}

  // ----------------------------------------------------------------------------
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
   * \param w_step pointer increment to get to next pixel column
   * \param h_step pointer increment to get to next pixel row
   * \param d_step pointer increment to get to next image channel
   */
  image_of( const T* first_pixel, size_t width, size_t height, size_t depth,
            ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step )
  : image( first_pixel, width, height, depth,
           w_step, h_step, d_step, image_pixel_traits_of<T>() ) {}

  // ----------------------------------------------------------------------------
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
   * \param w_step pointer increment to get to next pixel column
   * \param h_step pointer increment to get to next pixel row
   * \param d_step pointer increment to get to next image channel
   */
  image_of( const image_memory_sptr& mem,
            const T* first_pixel, size_t width, size_t height, size_t depth,
            ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step )
  : image( mem, first_pixel, width, height, depth,
           w_step, h_step, d_step, image_pixel_traits_of<T>() ) {}

  // ----------------------------------------------------------------------------
  /// Constructor from base class
  /**
   * The new image will share the same memory as the old image
   * \param other The other image.
   */
  explicit image_of( const image& other )
  : image(other)
  {
    if ( other.pixel_traits() != image_pixel_traits_of<T>() )
    {
      VITAL_THROW( image_type_mismatch_exception,
                   "kwiver::vital::image_of<T>(kwiver::vital::image)");
    }
  }

  // ----------------------------------------------------------------------------
  /// Assignment operator
  const image_of<T>& operator=( const image& other )
  {
    if ( other.pixel_traits() != image_pixel_traits_of<T>() )
    {
      VITAL_THROW( image_type_mismatch_exception,
                   "kwiver::vital::image_of<T>::operator=(kwiver::vital::image)");
    }
    image::operator=(other);
    return *this;
  }

  // ----------------------------------------------------------------------------
  /// Const access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a large image memory chunk.
   */
  const T* first_pixel() const { return reinterpret_cast<const T*>(first_pixel_); }

  // ----------------------------------------------------------------------------
  /// Access to the pointer to first image pixel
  /**
   * This may differ from \a data() if the image is a
   * window into a larger image memory chunk.
   */
  T* first_pixel() { return reinterpret_cast<T*>(first_pixel_); }

  // ----------------------------------------------------------------------------
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

    T const* data = this->first_pixel();
    if ( depth_ < 3 )
    {
      auto const v = data[w_step_ * i + h_step_ * j];
      return { v, v, v };
    }

    auto const r = data[w_step_ * i + h_step_ * j + d_step_ * 0];
    auto const g = data[w_step_ * i + h_step_ * j + d_step_ * 1];
    auto const b = data[w_step_ * i + h_step_ * j + d_step_ * 2];
    return { r, g, b };
  }

  // ----------------------------------------------------------------------------
  /// Access pixels in the first channel of the image
  /**
   * \param i width position (x)
   * \param j height position (y)
   */
  inline T& operator()( unsigned i, unsigned j )
  {
    return image::at<T>(i,j);
  }

  // ----------------------------------------------------------------------------
  /// Const access pixels in the first channel of the image
  inline const T& operator()( unsigned i, unsigned j ) const
  {
    return image::at<T>(i,j);
  }

  // ----------------------------------------------------------------------------
  /// Access pixels in the image (width, height, channel)
  inline T& operator()( unsigned i, unsigned j, unsigned k )
  {
    return image::at<T>(i,j,k);
  }

  // ----------------------------------------------------------------------------
  /// Const access pixels in the image (width, height, channel)
  inline const T& operator()( unsigned i, unsigned j, unsigned k ) const
  {
    return image::at<T>(i,j,k);
  }

};


/// Compare to images to see if the pixels have the same values.
/**
 * This does not require that the images have the same memory layout,
 * only that the images have the same dimensions and pixel values.
 * \param img1 first image to compare
 * \param img2 second image to compare
 */
VITAL_EXPORT bool equal_content( const image& img1, const image& img2 );

} }   // end namespace vital


#endif // VITAL_IMAGE_H_
