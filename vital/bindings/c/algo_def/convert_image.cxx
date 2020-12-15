// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to convert_image algorithm implementation
 */

#include "convert_image.h"

#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/image_container.h>

#include <vital/algo/convert_image.h>

DEFINE_COMMON_ALGO_API( convert_image );

/// Convert image base type
vital_image_container_t*
vital_algorithm_convert_image_convert( vital_algorithm_t *algo,
                                       vital_image_container_t *ic,
                                       vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::convert_image::convert", eh,
    kwiver::vital::image_container_sptr ic_sptr = kwiver::vital_c::IMGC_SPTR_CACHE.get( ic );
    kwiver::vital::algo::convert_image_sptr ci_sptr = kwiver::vital_c::ALGORITHM_convert_image_SPTR_CACHE.get( algo );
    kwiver::vital::image_container_sptr new_ic_sptr = ci_sptr->convert( ic_sptr );
    kwiver::vital_c::IMGC_SPTR_CACHE.store( new_ic_sptr );
    return reinterpret_cast<vital_image_container_t*>( new_ic_sptr.get() );
  );
  return 0;
}
