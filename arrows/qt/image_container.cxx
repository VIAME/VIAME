// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL image container implementation
 */

#include "image_container.h"

#include <vital/exceptions.h>
#include <vital/range/iota.h>

using kwiver::vital::range::iota;

namespace {

// ----------------------------------------------------------------------------
class qt_image_memory : public kwiver::vital::image_memory
{
public:
  qt_image_memory( QImage const& img ) : image_{ img }
  { size_ = static_cast< size_t >( image_.sizeInBytes() ); }

  virtual void* data() override { return image_.bits(); }

  QImage image_;
};

// ----------------------------------------------------------------------------
size_t
depth( QImage::Format format )
{
  switch( format )
  {
    case QImage::Format_Mono:
    case QImage::Format_MonoLSB:
    case QImage::Format_Grayscale8:
    case QImage::Format_Alpha8:
      return 1;
    case QImage::Format_Indexed8:
    case QImage::Format_RGB32:
    case QImage::Format_RGB16:
    case QImage::Format_RGB666:
    case QImage::Format_RGB555:
    case QImage::Format_RGB888:
    case QImage::Format_RGB444:
    case QImage::Format_RGBX8888:
    case QImage::Format_BGR30:
    case QImage::Format_RGB30:
      return 3;
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    case QImage::Format_ARGB8565_Premultiplied:
    case QImage::Format_ARGB6666_Premultiplied:
    case QImage::Format_ARGB8555_Premultiplied:
    case QImage::Format_ARGB4444_Premultiplied:
    case QImage::Format_RGBA8888:
    case QImage::Format_RGBA8888_Premultiplied:
    case QImage::Format_A2BGR30_Premultiplied:
    case QImage::Format_A2RGB30_Premultiplied:
      return 4;
    default:
      VITAL_THROW( kwiver::vital::image_type_mismatch_exception,
                   "depth: unsupported image format " +
                   std::to_string( format ) );
  }
}

// ----------------------------------------------------------------------------
uint8_t
get_pixel_gray( uint8_t const* in, ptrdiff_t pstep )
{
  Q_UNUSED( pstep );
  return *in;
}

// ----------------------------------------------------------------------------
uint32_t
get_pixel_rgb( uint8_t const* in, ptrdiff_t pstep )
{
  return qRgb( *( in + ( 0 * pstep ) ),
               *( in + ( 1 * pstep ) ),
               *( in + ( 2 * pstep ) ) );
}

// ----------------------------------------------------------------------------
uint32_t
get_pixel_rgba( uint8_t const* in, ptrdiff_t pstep )
{
  return qRgba( *( in + ( 0 * pstep ) ),
                *( in + ( 1 * pstep ) ),
                *( in + ( 2 * pstep ) ),
                *( in + ( 3 * pstep ) ) );
}

// ----------------------------------------------------------------------------
template < typename T >
QImage
vital_to_qt( kwiver::vital::image const& img, QImage::Format format,
             T ( * get_pixel )( uint8_t const*, ptrdiff_t ) )
{
  auto const w = static_cast< int >( img.width() );
  auto const h = static_cast< int >( img.height() );
  auto const pstep = img.d_step();

  auto* const in = reinterpret_cast< uint8_t const* >( img.first_pixel() );
  auto out = QImage{ w, h, format };

  // Iterate over scanlines
  for( auto const j : iota( h ) )
  {
    auto* const inl = in + ( j * img.h_step() );
    auto* const outl = reinterpret_cast< T* >( out.scanLine( j ) );

    // Iterate over pixels in scanline
    for( auto const i : iota( w ) )
    {
      auto* const inp = inl + ( i * img.w_step() );
      *( outl + i ) = get_pixel( inp, pstep );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
kwiver::vital::image
qt_to_vital( QImage const& img, QImage::Format format )
{
  return kwiver::arrows::qt::image_container::qt_to_vital(
    img.convertToFormat( format ) );
}

// ----------------------------------------------------------------------------
kwiver::vital::image
qt_to_vital_mono( QImage const& in )
{
  auto const w = static_cast< size_t >( in.width() );
  auto const h = static_cast< size_t >( in.height() );
  kwiver::vital::image_of< bool > out{ w, h };

  auto* const outp = out.first_pixel();
  for( auto const j : iota( in.height() ) )
  {
    auto* const outs = outp + ( j * static_cast< ptrdiff_t >( w ) );
    for( auto const i : iota( in.width() ) )
    {
      outs[ i ] = !!( in.pixel( i, j ) & 0xffffff );
    }
  }

  return out;
}

} // end namespace (anonymous)

namespace kwiver {

namespace arrows {

namespace qt {

// ----------------------------------------------------------------------------
image_container
::image_container( vital::image_container const& container )
{
  const qt::image_container* qic =
    dynamic_cast< qt::image_container const* >( &container );
  if( qic )
  {
    this->data_ = qic->data_;
  }
  else
  {
    this->data_ = vital_to_qt( container.get_image() );
  }
}

// ----------------------------------------------------------------------------
size_t
image_container
::depth() const
{
  return ::depth( data_.format() );
}

// ----------------------------------------------------------------------------
vital::image
image_container
::qt_to_vital( QImage const& img )
{
  if( img.isNull() )
  {
    return {};
  }

  auto const f = img.format();
  switch( f )
  {
    case QImage::Format_RGB888:
    case QImage::Format_RGBX8888:
    case QImage::Format_RGBA8888:
    {
      auto memory = std::make_shared< qt_image_memory >( img );
      return vital::image_of< uint8_t >(
        memory, reinterpret_cast< uint8_t* >( memory->data() ),
        static_cast< size_t >( img.width() ),
        static_cast< size_t >( img.height() ),
        ::depth( f ), img.depth() / 8, img.bytesPerLine(), 1 );
    }
    case QImage::Format_RGB32:
    {
      auto memory = std::make_shared< qt_image_memory >( img );
      return vital::image_of< uint8_t >(
        memory, reinterpret_cast< uint8_t* >( memory->data() ) + 2,
        static_cast< size_t >( img.width() ),
        static_cast< size_t >( img.height() ),
        ::depth( f ), img.depth() / 8, img.bytesPerLine(), -1 );
    }
    case QImage::Format_Grayscale8:
    case QImage::Format_Alpha8:
    {
      auto memory = std::make_shared< qt_image_memory >( img );
      return vital::image_of< uint8_t >(
        memory, reinterpret_cast< uint8_t* >( memory->data() ),
        static_cast< size_t >( img.width() ),
        static_cast< size_t >( img.height() ),
        1, 1, img.bytesPerLine(), 0 );
    }
    case QImage::Format_Mono:
    case QImage::Format_MonoLSB:
      return ::qt_to_vital_mono( img );
    default:
      switch( ::depth( f ) )
      {
        case 1:
          return ::qt_to_vital( img, QImage::Format_Grayscale8 );
        case 3:
          return ::qt_to_vital( img, QImage::Format_RGB888 );
        case 4:
          return ::qt_to_vital( img, QImage::Format_RGBA8888 );
        default:
          break;
      }
      break;
  }

  VITAL_THROW( kwiver::vital::image_type_mismatch_exception,
               "qt_to_vital: unsupported image format " +
               std::to_string( img.format() ) );
}

// ----------------------------------------------------------------------------
QImage
image_container
::vital_to_qt( vital::image const& img )
{
  constexpr auto MAX_DIM =
    static_cast< size_t >( std::numeric_limits< int >::max() );

  if( img.width() > MAX_DIM ||
      img.height() > MAX_DIM )
  {
    VITAL_THROW( kwiver::vital::image_exception,
                 "vital_to_qt: input image dimensions "
                 "(width = " + std::to_string( img.width() ) +
                 ", height = " + std::to_string( img.height() ) +
                 ") exceed maximum output dimension (" +
                 std::to_string( MAX_DIM ) + ")" );
  }

  auto const& pt = img.pixel_traits();
  if( pt.num_bytes == 1 )
  {
    if( pt.type == vital::image_pixel_traits::BOOL )
    {
      if( img.depth() != 1 )
      {
        VITAL_THROW( kwiver::vital::image_type_mismatch_exception,
                     "vital_to_qt: unsupported image format "
                     "(depth = " + std::to_string( img.depth() ) + ")" );
      }

      auto const w = static_cast< int >( img.width() );
      auto const h = static_cast< int >( img.height() );

      auto* const in = reinterpret_cast< bool const* >( img.first_pixel() );
      auto out = QImage{ w, h, QImage::Format_Mono };

      // Iterate over scanlines
      for( auto const j : iota( h ) )
      {
        auto* const inl = in + ( j * img.h_step() );
        auto* const outl = out.scanLine( j );

        uint8_t bits = 0;
        uint8_t mask = 1 << 7;

        // Iterate over pixels in scanline
        for( auto const i : iota( w ) )
        {
          // "Blend" bit into current byte
          auto* const inp = inl + ( i * img.w_step() );
          bits |= ( *inp ? mask : 0 );
          mask >>= 1;

          // When byte is full, write to output image
          if( !mask )
          {
            *( outl + ( i / 8 ) ) = bits;
            bits = 0;
            mask = 1 << 7;
          }
        }

        if( mask != 1 << 7 )
        {
          // Write last bits
          *( outl + ( w / 8 ) ) = bits;
        }
      }

      return out;
    }

    if( pt.type == vital::image_pixel_traits::UNSIGNED )
    {
      switch( img.depth() )
      {
        case 1:
          return ::vital_to_qt< uint8_t >(
            img, QImage::Format_Grayscale8, &get_pixel_gray );
        case 3:
          return ::vital_to_qt< uint32_t >(
            img, QImage::Format_RGB32, &get_pixel_rgb );
        case 4:
          return ::vital_to_qt< uint32_t >(
            img, QImage::Format_ARGB32, &get_pixel_rgba );
        default:
          VITAL_THROW( kwiver::vital::image_type_mismatch_exception,
                       "vital_to_qt: unsupported image format "
                       "(depth = " + std::to_string( img.depth() ) + ")" );
      }
    }
  }

  VITAL_THROW( kwiver::vital::image_type_mismatch_exception,
               "vital_to_qt: unsupported image format "
               "(type = " + std::to_string( pt.type ) +
               ", bytes = " + std::to_string( pt.num_bytes ) + ")" );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
