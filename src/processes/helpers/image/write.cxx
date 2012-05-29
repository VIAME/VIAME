/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "write.h"

#include "macros.h"

#include <vistk/pipeline/datum.h>

#include <vil/vil_image_view.h>
#include <vil/vil_save.h>

#include <string>

/**
 * \file write.cxx
 *
 * \brief Implementations of functions to help write images within the pipeline
 */

namespace vistk
{

template <typename PixType>
static void write(path_t const& path, datum_t const& dat);

write_func_t
write_for_pixtype(pixtype_t const& pixtype)
{
  SPECIFY_FUNCTION(write)

  return write_func_t();
}

template <typename PixType>
void
write(path_t const& path, datum_t const& dat)
{
  typedef vil_image_view<PixType> image_t;

  path_t::string_type const fpath = path.native();
  std::string const str(fpath.begin(), fpath.end());

  image_t const img = dat->get_datum<image_t>();

  bool const succeed = vil_save(img, str.c_str());

  if (!succeed)
  {
    /// \todo Log error.
  }
}

}
