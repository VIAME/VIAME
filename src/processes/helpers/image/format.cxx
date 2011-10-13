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

template <typename PixType>
class image_helper
{
  public:
    template <bool Grayscale = false, bool Alpha = false>
    struct port_types
    {
      public:
        static process::port_type_t const type;
    };
};

template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<false, true>::type = image_types::t_byte_rgb;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<false, false>::type = image_types::t_byte_rgb;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<true, true>::type = image_types::t_byte_grayscale;
template <> template <>
process::port_type_t const image_helper<uint8_t>::port_types<true, false>::type = image_types::t_byte_grayscale;

template <> template <>
process::port_type_t const image_helper<float>::port_types<false, true>::type = image_types::t_float_rgb;
template <> template <>
process::port_type_t const image_helper<float>::port_types<false, false>::type = image_types::t_float_rgb;
template <> template <>
process::port_type_t const image_helper<float>::port_types<true, true>::type = image_types::t_float_grayscale;
template <> template <>
process::port_type_t const image_helper<float>::port_types<true, false>::type = image_types::t_float_grayscale;

template <typename PixType> template <bool Grayscale, bool Alpha>
process::port_type_t const image_helper<PixType>::port_types<Grayscale, Alpha>::type = process::type_none;

}

process::port_type_t
port_type_for_pixtype(pixtype_t const& pixtype, bool grayscale, bool /*alpha*/)
{
  /// \todo Handle alpha parameter.

  if (pixtype == pixtypes::pixtype_byte())
  {
    if (grayscale)
    {
      return image_helper<uint8_t>::port_types<true>::type;
    }
    else
    {
      return image_helper<uint8_t>::port_types<false>::type;
    }
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    if (grayscale)
    {
      return image_helper<float>::port_types<true>::type;
    }
    else
    {
      return image_helper<float>::port_types<false>::type;
    }
  }

  return process::type_none;
}

}
