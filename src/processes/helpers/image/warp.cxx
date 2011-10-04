/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "warp.h"

#include <vistk/image/warp_image.h>

#include <vistk/utilities/homography.h>

#include <vistk/pipeline/datum.h>

#include <boost/cstdint.hpp>

#include <vil/vil_image_view.h>

/**
 * \file warp.cxx
 *
 * \brief Implementations of functions to help warp images within the pipeline.
 */

namespace vistk
{

template <typename PixType>
static datum_t warp(datum_t const& dat, datum_t const& trans_dat);

warp_func_t
warp_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte())
  {
    return &warp<uint8_t>;
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    return &warp<float>;
  }

  return warp_func_t();
}

template <typename PixType>
datum_t
warp(datum_t const& dat, datum_t const& trans_dat)
{
  typedef vil_image_view<PixType> image_t;
  typedef homography_base::transform_t transform_t;
  typedef warp_image<PixType> warp_t;

  image_t const image = dat->get_datum<image_t>();
  transform_t const transform = trans_dat->get_datum<transform_t>();

  warp_t const warp_img(0, 0, 1);

  image_t const warped_image = warp_img(image, transform);

  return datum::new_datum(warped_image);
}

}
