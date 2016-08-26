/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Interface to bounding box utilities
 */

#ifndef ARROWS_VXL_BOUNDING_BOX_H
#define ARROWS_VXL_BOUNDING_BOX_H

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vgl/vgl_box_2d.h>
#include <vital/types/bounding_box.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/**
 * @brief Convert vgl_box_2d to bounding_box
 *
 * This operator converts a vgl_box_2d to a kwiver bounding box.
 *
 * @param vbox vgl_box_2d to convert
 *
 * @return Equivalent bounding box.
 */
template <typename T>
KWIVER_ALGO_VXL_EXPORT
kwiver::vital::bounding_box<T> convert( const vgl_box_2d<T>& vbox )
{
  return kwiver::vital::bounding_box<T>( vbox.min_x(),
                                         vbox.min_y(),
                                         vbox.max_x(),
                                         vbox.max_y() );
}


// ------------------------------------------------------------------
/**
 * @brief Convert bounding box to vgl_box_2d
 *
 * @param bbox Bounding box to convert
 *
 * @return Equivalent vgl_box_2d
 */
template <typename T>
KWIVER_ALGO_VXL_EXPORT
vgl_box_2d<T> convert(const kwiver::vital::bounding_box<T>& bbox )
{
  return vgl_box_2d<T>( bbox.min_x(), bbox.max_x(),
                        bbox.min_y(), bbox.max_y() );
}

} } } // end namespace

#endif /* ARROWS_VXL_BOUNDING_BOX_H */
