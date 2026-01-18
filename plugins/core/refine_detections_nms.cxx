/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "refine_detections_nms.h"

namespace viame {

namespace kv = kwiver::vital;

// -----------------------------------------------------------------------------
void
refine_detections_nms
::initialize()
{
  // Compute min_scale_difference from max_scale_difference
  if( c_max_scale_difference != 0 )
  {
    if( c_max_scale_difference < 1.0 )
    {
      m_min_scale_difference = c_max_scale_difference;
    }
    else
    {
      m_min_scale_difference = 1.0 / c_max_scale_difference;
    }
  }
  else
  {
    m_min_scale_difference = 0.0;
  }
}

/// Check that the algorithm's currently configuration is valid
bool
refine_detections_nms
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
refine_detections_nms
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  // Returns detections sorted by confidence threshold
  kv::detected_object_set_sptr dets = detections->clone()->select();

  kv::detected_object_set_sptr results( new kv::detected_object_set() );

  // Prune first
  for(auto det : *dets)
  {
    bool should_add = true;

    kv::bounding_box_d det_bbox =
      scale_about_center( det->bounding_box(), c_nms_scale_factor );

    for( auto result : *results )
    {
      kv::bounding_box_d res_bbox =
        scale_about_center( result->bounding_box(), c_nms_scale_factor );

      kv::bounding_box_d overlap =
        kv::intersection( det_bbox, res_bbox );

      // Check how much they overlap. Only keep if the overlapped percent isn't too high
      if(overlap.min_x() < overlap.max_x() && overlap.min_y() < overlap.max_y() &&
         (overlap.area() / std::min(det_bbox.area(), res_bbox.area())) > c_max_overlap)
      {
        if(c_max_scale_difference == 0 ) // disabled
        {
          should_add = false;
          break;
        }
        else
        {
          double area_ratio = det_bbox.area() / res_bbox.area();

          if(area_ratio >= m_min_scale_difference && area_ratio <= c_max_scale_difference)
          {
            should_add = false;
            break;
          }
        }
      }
    }
    if(should_add) // It doesn't overlap too much, add it in
    {
      if(c_output_scale_factor != 1.0)
      {
        kv::bounding_box_d adj_bbox =
        scale_about_center( det->bounding_box(), c_output_scale_factor );

        det->set_bounding_box( adj_bbox );
      }
      results->add(det);
    }
  }

  return results;
}

} // end namespace viame
