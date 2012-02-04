/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "format.h"

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

#define PORT_TYPE(tpe, tname, fmt) \
  template <> template <>          \
  process::port_type_t const image_helper<tpe>::port_types<pix_##fmt>::type = "image/vil/" tname "/" #fmt

#define FORMAT_TYPES(tpe, tname) \
  PORT_TYPE(tpe, tname, rgb);    \
  PORT_TYPE(tpe, tname, bgr);    \
  PORT_TYPE(tpe, tname, rgba);   \
  PORT_TYPE(tpe, tname, bgra);   \
  PORT_TYPE(tpe, tname, yuv);    \
  PORT_TYPE(tpe, tname, gray)

FORMAT_TYPES(uint8_t, "byte");
FORMAT_TYPES(float, "float");

#undef FORMAT_TYPES
#undef PORT_TYPE

template <typename PixType> template <pixel_format_t Format>
process::port_type_t const image_helper<PixType>::port_types<Format>::type = process::type_none;

}

/**
 * \def PORT_TYPE
 *
 * \brief Checks for a pixel format.
 *
 * \param ptype The C++ pixel type.
 * \param fmt The pixel format.
 */
#define PORT_TYPE(ptype, fmt)                                \
  if (format == pixfmts::pixfmt_##fmt())                     \
  {                                                          \
    return image_helper<ptype>::port_types<pix_##fmt>::type; \
  }
/**
 * \def PORT_TYPES
 *
 * \brief Checks all pixel formats to get the port type.
 *
 * \param ptype The C++ pixel type.
 */
#define PORT_TYPES(ptype)     \
  PORT_TYPE(ptype, rgb)       \
  else PORT_TYPE(ptype, bgr)  \
  else PORT_TYPE(ptype, rgba) \
  else PORT_TYPE(ptype, bgra) \
  else PORT_TYPE(ptype, yuv)  \
  else PORT_TYPE(ptype, gray)

process::port_type_t
port_type_for_pixtype(pixtype_t const& pixtype, pixfmt_t const& format)
{
  if (pixtype == pixtypes::pixtype_byte())
  {
    PORT_TYPES(uint8_t)
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    PORT_TYPES(float)
  }

  return process::type_none;
}

#undef PORT_TYPES
#undef PORT_TYPE

}
