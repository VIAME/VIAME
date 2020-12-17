// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to image_io algorithm implementation
 */

#include "image_io.h"

#include <vital/algo/image_io.h>
#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/image_container.h>

DEFINE_COMMON_ALGO_API( image_io );

/// Load image from file
vital_image_container_t* vital_algorithm_image_io_load( vital_algorithm_t *algo,
                                                        char const *filename,
                                                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::image_io::load", eh,
    kwiver::vital::image_container_sptr ic_sptr = kwiver::vital_c::ALGORITHM_image_io_SPTR_CACHE.get( algo )->load( filename );
    kwiver::vital_c::IMGC_SPTR_CACHE.store( ic_sptr );
    return reinterpret_cast<vital_image_container_t*>( ic_sptr.get() );
  );
  return 0;
}

/// Save an image to file
void vital_algorithm_image_io_save( vital_algorithm_t *algo,
                                    char const *filename,
                                    vital_image_container_t *ic,
                                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::image_io::save", eh,
    kwiver::vital::image_container_sptr ic_sptr = kwiver::vital_c::IMGC_SPTR_CACHE.get(ic);
    kwiver::vital_c::ALGORITHM_image_io_SPTR_CACHE.get( algo )->save( filename, ic_sptr );
  );
}
