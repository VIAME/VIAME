/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "detected_object_set_util.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
void
scale_detections( detected_object_set_sptr dos,
       double scale_factor )
{
  if( scale_factor == 1.0 )
  {
    return;
  }

  for( auto detection : *dos )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::scale( bbox, scale_factor );
    detection->set_bounding_box( bbox );
  }
}

// ------------------------------------------------------------------
void
shift_detections( detected_object_set_sptr dos,
       double col_shift, double row_shift )
{
  if( col_shift == 0.0 && row_shift == 0.0 )
  {
    return;
  }

  for( auto detection : *dos )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::translate( bbox,
      bounding_box_d::vector_type( col_shift, row_shift ) );
    detection->set_bounding_box( bbox );
  }
}

} } // end namespace
