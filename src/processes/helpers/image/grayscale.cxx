/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "grayscale.h"

#include <vistk/pipeline/datum.h>

#include <boost/cstdint.hpp>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>

/**
 * \file grayscale.cxx
 *
 * \brief Implementations of functions to help convert images to grayscale in the pipeline.
 */

namespace vistk
{

template <typename PixType>
static datum_t convert_to_gray(datum_t const& dat);

gray_func_t
gray_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte())
  {
    return &convert_to_gray<uint8_t>;
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    return &convert_to_gray<float>;
  }

  return gray_func_t();
}

template <typename PixType>
datum_t
convert_to_gray(datum_t const& dat)
{
  typedef vil_image_view<PixType> image_t;

  image_t const rgb_image = dat->get_datum<image_t>();

  if (rgb_image.nplanes() == 1)
  {
    return datum::new_datum(rgb_image);
  }

  if (rgb_image.nplanes() != 3)
  {
    static std::string const err_string = "Input image does not have three planes.";

    return datum::error_datum(err_string);
  }

  image_t gray_image;

  vil_convert_planes_to_grey(rgb_image, gray_image);

  return datum::new_datum(gray_image);
}

}
