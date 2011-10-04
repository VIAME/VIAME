/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "crop.h"

#include <vistk/pipeline/datum.h>

#include <boost/cstdint.hpp>

#include <vil/vil_crop.h>
#include <vil/vil_image_view.h>

/**
 * \file crop.cxx
 *
 * \brief Implementations of functions to help cropping images in the pipeline.
 */

namespace vistk
{

template <typename PixType>
static datum_t crop(datum_t const& dat, size_t x_offset, size_t y_offset, size_t width, size_t height);

crop_func_t
crop_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte())
  {
    return &crop<uint8_t>;
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    return &crop<float>;
  }

  return crop_func_t();
}

template <typename PixType>
datum_t
crop(datum_t const& dat, size_t x_offset, size_t y_offset, size_t width, size_t height)
{
  typedef vil_image_view<PixType> image_t;

  image_t const img = dat->get_datum<image_t>();

  /// \todo Sanity check the parameters.

  image_t const crop_img = vil_crop(img, x_offset, width, y_offset, height);

  if (!crop_img)
  {
    return datum::error_datum("Unable to crop the image.");
  }

  return datum::new_datum(crop_img);
}

}
