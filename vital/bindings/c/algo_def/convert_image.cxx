/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
