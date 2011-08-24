/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "vil_helper.h"

#include <vistk/pipeline_types/image_types.h>

#include <vil/vil_convert.h>
#include <vil/vil_load.h>
#include <vil/vil_save.h>

namespace vistk
{

pixtype_t const pixtypes::pixtype_byte = pixtype_t("byte");
pixtype_t const pixtypes::pixtype_float = pixtype_t("float");

template <> template <>
process::port_type_t const vil_helper<uint8_t>::port_types<true, true>::type = image_types::t_byte_rgb;
template <> template <>
process::port_type_t const vil_helper<uint8_t>::port_types<true, false>::type = image_types::t_byte_rgb;
template <> template <>
process::port_type_t const vil_helper<uint8_t>::port_types<false, true>::type = image_types::t_byte_grayscale;
template <> template <>
process::port_type_t const vil_helper<uint8_t>::port_types<false, false>::type = image_types::t_byte_grayscale;

template <> template <>
process::port_type_t const vil_helper<float>::port_types<true, true>::type = image_types::t_float_rgb;
template <> template <>
process::port_type_t const vil_helper<float>::port_types<true, false>::type = image_types::t_float_rgb;
template <> template <>
process::port_type_t const vil_helper<float>::port_types<false, true>::type = image_types::t_float_grayscale;
template <> template <>
process::port_type_t const vil_helper<float>::port_types<false, false>::type = image_types::t_float_grayscale;

template <class PixType> template <bool Grayscale, bool Alpha>
process::port_type_t const vil_helper<PixType>::port_types<Grayscale, Alpha>::type = process::type_none;

template <class PixType>
datum_t
vil_helper<PixType>
::read(path_t const& path)
{
  path_t::string_type const fstr = path.native();
  std::string const str(fstr.begin(), fstr.end());

  image_t img = vil_load(str.c_str());

  if (!img)
  {
    return datum::error_datum("Unable to load image.");
  }

  return datum::new_datum(img);
}

template <class PixType>
void
vil_helper<PixType>
::write(path_t const& path, datum_t const& dat)
{
  path_t::string_type const fpath = path.native();
  std::string const str(fpath.begin(), fpath.end());

  image_t const img = dat->get_datum<image_t>();

  bool const succeed = vil_save(img, str.c_str());

  if (!succeed)
  {
    /// \todo Log error.
  }
}

template <class PixType>
datum_t
vil_helper<PixType>
::convert_to_gray(datum_t const& dat)
{
  image_t const rgb_image = dat->get_datum<image_t>();

  if (rgb_image.nplanes() == 1)
  {
    return datum::new_datum(rgb_image);
  }

  if (rgb_image.nplanes() != 3)
  {
    return datum::error_datum("Input image does not have three planes.");
  }

  image_t gray_image;

  vil_convert_planes_to_grey(rgb_image, gray_image);

  return datum::new_datum(gray_image);
}

process::port_type_t
port_type_for_pixtype(pixtype_t const& pixtype, bool grayscale, bool /*alpha*/)
{
  /// \todo Handle alpha parameter.

  if (pixtype == pixtypes::pixtype_byte)
  {
    if (grayscale)
    {
      return vil_helper<uint8_t>::port_types<true>::type;
    }
    else
    {
      return vil_helper<uint8_t>::port_types<false>::type;
    }
  }
  else if (pixtype == pixtypes::pixtypes::pixtype_float)
  {
    if (grayscale)
    {
      return vil_helper<float>::port_types<true>::type;
    }
    else
    {
      return vil_helper<float>::port_types<false>::type;
    }
  }

  return process::type_none;
}

read_func_t
read_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte)
  {
    return vil_helper<uint8_t>::read;
  }
  else if (pixtype == pixtypes::pixtype_float)
  {
    return vil_helper<float>::read;
  }

  return NULL;
}

write_func_t
write_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte)
  {
    return vil_helper<uint8_t>::write;
  }
  else if (pixtype == pixtypes::pixtype_float)
  {
    return vil_helper<float>::write;
  }

  return NULL;
}

gray_func_t
gray_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte)
  {
    return vil_helper<uint8_t>::convert_to_gray;
  }
  else if (pixtype == pixtypes::pixtype_float)
  {
    return vil_helper<float>::convert_to_gray;
  }

  return NULL;
}

}
