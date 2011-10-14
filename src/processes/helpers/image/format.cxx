/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "format.h"

#include <vistk/pipeline_types/image_types.h>

#include <boost/cstdint.hpp>

/**
 * \file format.cxx
 *
 * \brief Implementations of functions to help manage image formats in the pipeline.
 */

namespace vistk
{

namespace
{

enum pixel_format_t
{
  pix_rgb,
  pix_bgr,
  pix_rgba,
  pix_bgra,
  pix_yuv,
  pix_gray
};

template <typename PixType>
class image_helper
{
  public:
    template <pixel_format_t Format>
    struct port_types
    {
      public:
        static process::port_type_t const type;
    };
};

template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_rgb>::type = image_types::t_byte_rgb;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_bgr>::type = image_types::t_byte_bgr;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_rgba>::type = image_types::t_byte_rgba;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_bgra>::type = image_types::t_byte_bgra;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_yuv>::type = image_types::t_byte_grayscale;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<pix_gray>::type = image_types::t_byte_grayscale;

template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_rgb>::type = image_types::t_float_rgb;
template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_bgr>::type = image_types::t_float_bgr;
template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_rgba>::type = image_types::t_float_rgba;
template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_bgra>::type = image_types::t_float_bgra;
template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_yuv>::type = image_types::t_float_grayscale;
template <> template <>
process::port_type_t const image_helper<float>::port_types<pix_gray>::type = image_types::t_float_grayscale;

template <typename PixType> template <pixel_format_t Format>
process::port_type_t const image_helper<PixType>::port_types<Format>::type = process::type_none;

}

process::port_type_t
port_type_for_pixtype(pixtype_t const& pixtype, pixfmt_t const& format)
{
#define PORT_TYPE(ptype, pix) \
  if (format == pixfmts::pixfmt_##pix())                     \
  {                                                          \
    return image_helper<ptype>::port_types<pix_##pix>::type; \
  }
#define PORT_TYPES(ptype)     \
  PORT_TYPE(ptype, rgb)       \
  else PORT_TYPE(ptype, bgr)  \
  else PORT_TYPE(ptype, rgba) \
  else PORT_TYPE(ptype, bgra) \
  else PORT_TYPE(ptype, yuv)  \
  else PORT_TYPE(ptype, gray)

  if (pixtype == pixtypes::pixtype_byte())
  {
    PORT_TYPES(uint8_t)
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    PORT_TYPES(float)
  }
#undef PORT_TYPES
#undef PORT_TYPE

  return process::type_none;
}

}
