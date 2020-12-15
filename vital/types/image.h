// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

// ===========================================================================
/// The representation of an in-memory image.
/**
 * This base image class represents an image with a dynamic data type.  The
 * underlying data type can be queried using pixel_traits().  To properly
 * access individual pixels the data type must be known.  The templated at<T>()
 * member function provides direct access to pixels.  Alternatively, cast the
 * image itself into an image_of object.  The typed image_of class is a bit
 * easier to work with once the type is known, but this base class is useful
 * in APIs that may operate on images of various types.
 *
 * Memory Management
 * -----------------
 *
 * This image class supports two modes of memory management.  Either the image
 * owns its memory or it does not.  If the image owns its memory the
 * image::memory() function will return a shared pointer to that image_memory
 * object.  Otherwise, image::memory() will return nullptr.  In both cases,
 * image::first_pixel() returns a pointer to the first pixel of the memory
 * that appears in the image.  The address of the first pixel does not need
 * to match the starting address of the image_memory.  There can be multiple
 * different views into the same memory (e.g. a cropped image view) and these
 * views may use the same memory object with a different offsets to the first
 * pixel, size, and step parameters.
 *
 * Typically the image manages its own memory in a reference counted
 * image_memory object.  Creating a new image will allocate this memory, which
 * can be accessed from image::memory().  Copying an image will make a shallow
 * copy refering to the same memory object, and the memory will be deleted
 * when all images are done with it.
 *
 * There is a special constructor that allows construction of an image as a
 * veiw into some existing memory.  In this case the image does not own the
 * memory and image::memory() will return nullptr.  The user must ensure that
 * the memory exists for the lifetime of the image.
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
  /**
   * \copydoc image::memory()
   */
  const image_memory_sptr& memory() const { return data_; }

  /// Access to the image memory
  /**
   * In most cases, when interacting with image data, you should use the
   * first_pixel() function instead of memory().  The memory() function
   * provides access to the underlying reference counted memory for advanced
   * memory management applications.  It returns a block of data that contains
   * the image somewhere within.  The block of data may contain only the
   * pixels in this image, but it could also contain much more hidden data
   * if the image is a crop or subsampling of an original image that was
   * larger.  The blocks of memory are typically shared between copies of
   * this image, and each copy may have a different view into the memory.
   *
   * This function may also return \c nullptr for a valid image that is a
   * view into some external memory (first_pixel() is still valid in this
   * case).  Use caution when accessing the memory directly and always check
   * that the memory is not \c nullptr.  Never assume that the image data
   * must be contained in the memory block returned by this function.
   */
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
   * \copydoc image::first_pixel()
   */
  const void* first_pixel() const { return first_pixel_; }

  /// Access to the pointer to first image pixel
  /**
   * Returns a raw void pointer to the first pixel in the image.
   * This is the starting point for iterating through the image using
   * offsets of w_step(), h_step(), and d_step().  See also
   * image_of::first_pixel() for a variant of this function that
   * returns a pointer to the underlying pixel type.
   *
   * \note the address returned may differ from the starting address
   * returned by image::memory() if the image is a window into a larger
   * block of image memory.
   *
   * \note If the address returned is not \c nullptr but image::memory()
   * returns \c nullptr, then this image is a view into external memory
   * not owned by this image object.
   *
   * \sa image_of::first_pixel()
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
   * Compares this image to another image to test equality.
   *
   * \param other image to compare with
   *
   * \note This function computes only "shallow" equality.  That is, the images
   *       are considered equal if they point to the same memory and have the
   *       dimensions and pixel step sizes.  Deep equality testing requires
   *       stepping through and testing that the values of each pixel are the
   *       same even if the memory and possibly memory layout differ.
   *
   * \sa   For deep equality comparison see equal_content
   */
  bool operator==( image const& other ) const;

  /// Inequality operator
  /**
  * Compares this image to another image to test inequality.
  *
  * \param other image to compare with
  *
  * \note This function computes only "shallow" inequality.  Refer to the
  *       equality operator (==) for details.
  */
  bool operator!=(image const& other) const
  {
    return !(*this == other);
  }

  /// Access pixels in the first channel of the image
  /**
   * \param i width position (x)
   * \param j height position (y)
   */
  template <typename T>
  inline T& at( size_t i, size_t j )
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(size_t, size_t)");
    }
    return reinterpret_cast<T*>(first_pixel_)[w_step_ * i + h_step_ * j];
  }

  /// Const access pixels in the first channel of the image
  template <typename T>
  inline const T& at( size_t i, size_t j ) const
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(size_t, size_t) const");
    }
    return reinterpret_cast<const T*>(first_pixel_)[w_step_ * i + h_step_ * j];
  }

  /// Access pixels in the image (width, height, channel)
  template <typename T>
  inline T& at( size_t i, size_t j, size_t k )
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(size_t, size_t, size_t)");
    }
    return reinterpret_cast<T*>(first_pixel_)[w_step_ * i + h_step_ * j + d_step_ * k];
  }

  /// Const access pixels in the image (width, height, channel)
  template <typename T>
  inline const T& at( size_t i, size_t j, size_t k ) const
  {
    if( i >= width_ || j >= height_ || k >= depth_ )
    {
      throw std::out_of_range("kwiver::vital::image::at<T>(size_t, size_t, size_t) const");
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

  /// Get a cropped view of the image.
  /**
   * Get a cropped view of the image. The cropped view shares memory with the
   * original image so no deep copy is done.
   * \param x_offset start of the crop region in x (width)
   * \param y_offset start of the crop region in y (height)
   * \param width width of the crop region
   * \param height height of the crop region
   */
  image crop(size_t x_offset, size_t y_offset, size_t width, size_t height) const;

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

// ===========================================================================
/// The representation of a type-specific in-memory image.
/**
 * This class is derived from the image() class to provide convenience
 * functions that require the pixel type to be known at compile time.
 * This derived class does not add any data or change any behavior of the
 * base image() class.  It simply provides a strongly-typed view of the data.
 * The constructors in this class make it easier to construct an image.
 * For example,
\code
image I;
// direct construction of a double image
I = image(100, 100, 1, false, pixel_traits_of<double>());
// equivalent construction using image_of
I = image_of<double>(100, 100);
\endcode
 *
 * Once cast as an image_of() the operator()() is available to directly access
 * pixels with a simpler syntax. For example
\code
image_of<float> my_img(100, 100);     // make a float image of size 100 x 100
float val = my_img(10, 10);           // get pixel at 10, 10
      val = my_img.at<float>(10, 10); // image::at method does the same thing
\endcode
 *
 * An image() can be directly assigned to an image_of() object and this will
 * throw a image_type_mismatch_exception if the underlying type does not match.
 * For example
\code
// make a 16-bit unsigned image with the base class
image my_img(100, 100, 1, false, pixel_traits_of<uint16_t>());

image_of<uint16_t> my_img16 = my_img; // this works
image_of<float> my_imgf = my_img;     // this throws an exception
\endcode
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
  inline rgb_color at( size_t i, size_t j ) const
  {
    if( i >= width_ || j >= height_ )
    {
      throw std::out_of_range("kwiver::vital::image::at(size_t, size_t) const");
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
  inline T& operator()( size_t i, size_t j )
  {
    return image::at<T>(i,j);
  }

  // ----------------------------------------------------------------------------
  /// Const access pixels in the first channel of the image
  inline const T& operator()( size_t i, size_t j ) const
  {
    return image::at<T>(i,j);
  }

  // ----------------------------------------------------------------------------
  /// Access pixels in the image (width, height, channel)
  inline T& operator()( size_t i, size_t j, size_t k )
  {
    return image::at<T>(i,j,k);
  }

  // ----------------------------------------------------------------------------
  /// Const access pixels in the image (width, height, channel)
  inline const T& operator()( size_t i, size_t j, size_t k ) const
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
