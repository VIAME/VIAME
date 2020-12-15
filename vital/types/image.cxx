// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core image class implementation
 */

#include "image.h"
#include <cstring>
#include <utility>

namespace kwiver {
namespace vital {

template <typename T> VITAL_EXPORT
image_pixel_traits::pixel_type const image_pixel_traits_of<T>::static_type;

template struct image_pixel_traits_of<char>;
template struct image_pixel_traits_of<signed char>;
template struct image_pixel_traits_of<unsigned char>;
template struct image_pixel_traits_of<signed short>;
template struct image_pixel_traits_of<unsigned short>;
template struct image_pixel_traits_of<signed int>;
template struct image_pixel_traits_of<unsigned int>;
template struct image_pixel_traits_of<signed long>;
template struct image_pixel_traits_of<unsigned long>;
template struct image_pixel_traits_of<signed long long>;
template struct image_pixel_traits_of<unsigned long long>;
template struct image_pixel_traits_of<float>;
template struct image_pixel_traits_of<double>;

VITAL_EXPORT
image_pixel_traits::pixel_type const image_pixel_traits_of<bool>::static_type;

template <> struct image_pixel_traits_of<bool>;

/// Output stream operator for image_pixel_traits::pixel_type
std::ostream& operator<<(std::ostream& os, image_pixel_traits::pixel_type pt)
{
  switch (pt)
  {
    case image_pixel_traits::UNKNOWN:
      os << "Unknown";
      break;
    case image_pixel_traits::UNSIGNED:
      os << "Unsigned";
      break;
    case image_pixel_traits::SIGNED:
      os << "Signed";
      break;
    case image_pixel_traits::FLOAT:
      os << "Float";
      break;
    case image_pixel_traits::BOOL:
      os << "Bool";
      break;
    default:
      os << "Invalid";
      break;
  }
  return os;
}

/// Output stream operator for image_pixel_traits
std::ostream& operator<<(std::ostream& os, image_pixel_traits const& pt)
{
  os << pt.type <<"_"<<pt.num_bytes;
  return os;
}

/// Default Constructor
image_memory
::image_memory()
  : data_( 0 ),
    size_( 0 )
{
}

/// Constructor - allocated n bytes
image_memory
::image_memory( size_t n )
  : data_( new char[n] ),
    size_( n )
{
}

/// Copy Constructor
image_memory
::image_memory( const image_memory& other )
  : data_( new char[other.size()] ),
    size_( other.size() )
{
  std::memcpy( data_, other.data_, size_ );
}

/// Destructor
image_memory
::~image_memory()
{
  delete [] reinterpret_cast< char* > ( data_ );
}

/// Assignment operator
image_memory&
image_memory
::operator=( const image_memory& other )
{
  if ( this == &other )
  {
    return *this;
  }

  if ( size_ != other.size_ )
  {
    delete [] reinterpret_cast< char* > ( data_ );
    data_ = 0;
    if ( other.size_ > 0 )
    {
      data_ = new char[other.size_];
    }
    size_ = other.size_;
  }

  std::memcpy( data_, other.data_, size_ );
  return *this;
}

/// Return a pointer to the allocated memory
void*
image_memory
::data()
{
  return data_;
}

//======================================================================

/// Default Constructor
image
::image(const image_pixel_traits& pt)
  : data_(),
    first_pixel_( NULL ),
    pixel_traits_( pt ),
    width_( 0 ),
    height_( 0 ),
    depth_( 0 ),
    w_step_( 0 ),
    h_step_( 0 ),
    d_step_( 0 )
{
}

/// Constructor that allocates image memory
image
::image( size_t width, size_t height, size_t depth,
         bool interleave, const image_pixel_traits& pt)
  : data_( new image_memory( width * height * depth * pt.num_bytes) ),
    first_pixel_( data_->data() ),
    pixel_traits_( pt ),
    width_( width ),
    height_( height ),
    depth_( depth ),
    w_step_( 1 ),
    h_step_( width ),
    d_step_( width * height )
{
  if ( interleave )
  {
    d_step_ = 1;
    w_step_ = depth;
    h_step_ = depth * width;
  }
}

/// Constructor that points at existing memory
image
::image( const void* first_pixel,
         size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step,
         const image_pixel_traits& pt )
  : data_(),
    first_pixel_( const_cast< void* > ( first_pixel ) ),
    pixel_traits_( pt ),
    width_( width ),
    height_( height ),
    depth_( depth ),
    w_step_( w_step ),
    h_step_( h_step ),
    d_step_( d_step )
{
}

/// Constructor that shares memory with another image
image
::image( const image_memory_sptr& mem,
         const void* first_pixel,
         size_t width, size_t height, size_t depth,
         ptrdiff_t w_step, ptrdiff_t h_step, ptrdiff_t d_step,
         const image_pixel_traits& pt)
  : data_( mem ),
    first_pixel_( const_cast< void* > ( first_pixel ) ),
    pixel_traits_( pt ),
    width_( width ),
    height_( height ),
    depth_( depth ),
    w_step_( w_step ),
    h_step_( h_step ),
    d_step_( d_step )
{
}

/// Copy Constructor
image
::image( const image& other )
  : data_( other.data_ ),
    first_pixel_( other.first_pixel_ ),
    pixel_traits_( other.pixel_traits_ ),
    width_( other.width_ ),
    height_( other.height_ ),
    depth_( other.depth_ ),
    w_step_( other.w_step_ ),
    h_step_( other.h_step_ ),
    d_step_( other.d_step_ )
{
}

/// Assignment operator
const image&
image
::operator=( const image& other )
{
  if ( this == &other )
  {
    return *this;
  }
  data_         = other.data_;
  first_pixel_  = other.first_pixel_;
  pixel_traits_ = other.pixel_traits_;
  width_        = other.width_;
  height_       = other.height_;
  depth_        = other.depth_;
  w_step_       = other.w_step_;
  h_step_       = other.h_step_;
  d_step_       = other.d_step_;
  return *this;
}

/// Equality operator
bool image::
operator==( image const& other ) const
{
  return  data_         == other.data_ &&
          first_pixel_  == other.first_pixel_ &&
          pixel_traits_ == other.pixel_traits_ &&
          width_        == other.width_ &&
          height_       == other.height_ &&
          depth_        == other.depth_ &&
          w_step_       == other.w_step_ &&
          h_step_       == other.h_step_ &&
          d_step_       == other.d_step_;
}

/// The size of the image data in bytes
size_t
image
::size() const
{
  if ( ! data_ )
  {
    return 0;
  }
  return data_->size();
}

/// Return true if the pixels accessible in this image form a contiguous memory block
bool
image
::is_contiguous() const
{
  // sort the step/size pairs from smallest to largest step
  std::pair<ptrdiff_t, size_t> sizes[3] = { {w_step_, width_}, {h_step_, height_}, {d_step_, depth_} };
  if (sizes[0].first > sizes[1].first)
  {
    std::swap(sizes[0], sizes[1]);
  }
  if (sizes[0].first > sizes[2].first)
  {
    std::swap(sizes[0], sizes[2]);
  }
  if (sizes[1].first > sizes[2].first)
  {
    std::swap(sizes[1], sizes[2]);
  }
  return sizes[0].first == 1 &&
         sizes[1].first == static_cast<ptrdiff_t>(sizes[0].second) &&
         sizes[2].first == static_cast<ptrdiff_t>(sizes[0].second * sizes[1].second);
}

/// Deep copy the image data from another image into this one
void
image
::copy_from( const image& other )
{
  if ( pixel_traits_ != other.pixel_traits_ )
  {
    // clear the current image so that set_size will allocate a new one
    // with the correct pixel_traits_
    pixel_traits_ = other.pixel_traits_;
    width_ = 0;
    height_ = 0;
    depth_ = 0;
    data_.reset();
    first_pixel_ = NULL;
  }
  set_size( other.width_, other.height_, other.depth_ );

  const ptrdiff_t d_step = this->d_step_ * pixel_traits_.num_bytes;
  const ptrdiff_t h_step = this->h_step_ * pixel_traits_.num_bytes;
  const ptrdiff_t w_step = this->w_step_ * pixel_traits_.num_bytes;
  const ptrdiff_t o_d_step = other.d_step_ * pixel_traits_.num_bytes;
  const ptrdiff_t o_h_step = other.h_step_ * pixel_traits_.num_bytes;
  const ptrdiff_t o_w_step = other.w_step_ * pixel_traits_.num_bytes;

  // copy data a raw bytes regardless of underlying data type
  const byte* o_data = reinterpret_cast<const byte*>(other.first_pixel_);
  byte* data = reinterpret_cast<byte*>(this->first_pixel_);

  // if data are contiguous and laid out in the same order then use memcpy
  if ( d_step == o_d_step &&
       h_step == o_h_step &&
       w_step == o_w_step &&
       this->is_contiguous() )
  {
    std::memcpy(data, o_data, width_ * height_ * depth_ * pixel_traits_.num_bytes);
    return;
  }

  for ( unsigned int d = 0; d < depth_; ++d, o_data += o_d_step, data += d_step )
  {
    const byte* o_row = o_data;
    byte* row = data;
    for ( unsigned int h = 0; h < height_; ++h, o_row += o_h_step, row += h_step )
    {
      const byte* o_pixel = o_row;
      byte* pixel = row;
      for ( unsigned int w = 0; w < width_; ++w, o_pixel += o_w_step, pixel += w_step )
      {
        std::memcpy(pixel, o_pixel, pixel_traits_.num_bytes);
      }
    }
  }
}

/// Set the size of the image.
void
image
::set_size( size_t width, size_t height, size_t depth )
{
  if ( ( width == width_ ) && ( height == height_ ) && ( depth == depth_ ) )
  {
    return;
  }

  data_ = image_memory_sptr( new image_memory( width * height * depth * pixel_traits_.num_bytes ) );
  width_ = width;
  height_ = height;
  depth_ = depth;
  first_pixel_ = data_->data();

  // preserve the pixel ordering (e.g. interleaved) as much as possible
  if ( ( w_step_ == 0 ) || ( w_step_ != static_cast< ptrdiff_t > ( depth_ ) ) )
  {
    w_step_ = 1;
  }
  h_step_ = width * w_step_;
  d_step_ = ( w_step_ == 1 ) ? width * height : 1;
}

/// Get a cropped view of the image.
image
image
::crop(size_t x_offset, size_t y_offset, size_t width, size_t height) const
{
  auto crop_first_pixel = reinterpret_cast< const char* >( this->first_pixel() );
  crop_first_pixel +=
    static_cast< ptrdiff_t >( this->pixel_traits().num_bytes ) *
    ( this->w_step() * static_cast< ptrdiff_t >( x_offset ) +
      this->h_step() * static_cast< ptrdiff_t >( y_offset ) );
  return image( this->memory(), crop_first_pixel,
                width, height, this->depth(),
                this->w_step(), this->h_step(), this->d_step(),
                this->pixel_traits() );
}

/// Compare to images to see if the pixels have the same values.
bool
equal_content( const image& img1, const image& img2 )
{
  const size_t width = img1.width();
  const size_t height = img1.height();
  const size_t depth = img1.depth();
  const image_pixel_traits& pt = img1.pixel_traits();
  if ( ( width  != img2.width() ) ||
       ( height != img2.height() ) ||
       ( depth  != img2.depth() ) ||
       ( pt     != img2.pixel_traits() ) )
  {
    return false;
  }

  const ptrdiff_t ws1 = img1.w_step() * pt.num_bytes;
  const ptrdiff_t hs1 = img1.h_step() * pt.num_bytes;
  const ptrdiff_t ds1 = img1.d_step() * pt.num_bytes;
  const ptrdiff_t ws2 = img2.w_step() * pt.num_bytes;
  const ptrdiff_t hs2 = img2.h_step() * pt.num_bytes;
  const ptrdiff_t ds2 = img2.d_step() * pt.num_bytes;

  // test equality of data using bytes regardless of underlying data format
  const byte* plane1 = reinterpret_cast<const byte*>(img1.first_pixel());
  const byte* plane2 = reinterpret_cast<const byte*>(img2.first_pixel());
  for ( unsigned k = 0; k < img1.depth(); ++k, plane1+=ds1, plane2+=ds2 )
  {
    const byte* row1 = plane1;
    const byte* row2 = plane2;
    for ( unsigned j = 0; j < img1.height(); ++j, row1+=hs1, row2+=hs2 )
    {
      const byte* col1 = row1;
      const byte* col2 = row2;
      for ( unsigned i = 0; i < img1.width(); ++i, col1+=ws1, col2+=ws2 )
      {
        const byte* byte1 = col1;
        const byte* byte2 = col2;
        for ( unsigned b = 0; b < pt.num_bytes; ++b, ++byte1, ++byte2 )
        {
          if ( *byte1 != *byte2 )
          {
            return false;
          }
        }
      }
    }
  }
  return true;
}

}
}   // end namespace vital
