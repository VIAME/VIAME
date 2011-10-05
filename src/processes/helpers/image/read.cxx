/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "read.h"

#include <vistk/pipeline/datum.h>

#include <boost/cstdint.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_load.h>

#include <string>

/**
 * \file read.cxx
 *
 * \brief Implementations of functions to help read images within the pipeline
 */

namespace vistk
{

template <typename PixType>
static datum_t read(path_t const& path);

read_func_t
read_for_pixtype(pixtype_t const& pixtype)
{
  if (pixtype == pixtypes::pixtype_byte())
  {
    return &read<uint8_t>;
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    return &read<float>;
  }

  return NULL;
}

template <typename PixType>
datum_t
read(path_t const& path)
{
  typedef vil_image_view<PixType> image_t;

  path_t::string_type const fstr = path.native();
  std::string const str(fstr.begin(), fstr.end());

  image_t img = vil_load(str.c_str());

  if (!img)
  {
    static datum::error_t const err_string = datum::error_t("Unable to load image.");

    return datum::error_datum(err_string);
  }

  return datum::new_datum(img);
}

}
