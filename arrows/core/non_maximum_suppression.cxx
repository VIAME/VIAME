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

#include "non_maximum_suppression.h"

namespace kwiver {
namespace arrows {
namespace core {

/// Private implementation class
class non_maximum_suppression::priv
{
public:

  /// Constructor
  priv()
  : scale_factor( 1.0 ),
    max_overlap(0.8)
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  double scale_factor;
  double max_overlap;

};


/// Constructor
non_maximum_suppression
::non_maximum_suppression()
: d_( new priv() )
{
}


/// Destructor
non_maximum_suppression
::~non_maximum_suppression()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
non_maximum_suppression
::get_configuration() const
{
  vital::config_block_sptr config = vital::algo::refine_detections::get_configuration();

  config->set_value( "scale_factor", d_->scale_factor,
                     "The factor by which the refined detections are scaled." );

  config->set_value( "max_overlap", d_->max_overlap,
                     "The maximum percent a detection can overlap with another before it's discarded." );

  return config;
}


/// Set this algorithm's properties via a config block
void
non_maximum_suppression
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->scale_factor = config->get_value<double>( "scale_factor" );
  d_->max_overlap = config->get_value<double>( "max_overlap" );
}

/// Check that the algorithm's currently configuration is valid
bool
non_maximum_suppression
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

// ------------------------------------------------------------------
vital::detected_object_set_sptr
non_maximum_suppression
::refine( vital::image_container_sptr image_data,
          vital::detected_object_set_sptr detections ) const
{
  vital::detected_object_set_sptr dets = detections->clone()->select(); // sort by confidence threshold

  vital::detected_object_set_sptr results(new vital::detected_object_set());

  // Prune first
  for(auto det = dets->begin(); det != dets->end(); det++)
  {
    bool should_add = true;
    for(auto result = results->begin(); result != results->end(); result++)
    {
      kwiver::vital::bounding_box_d det_bbox = (*det)->bounding_box();
      kwiver::vital::bounding_box_d res_bbox = (*result)->bounding_box();
      kwiver::vital::bounding_box_d overlap = kwiver::vital::intersection(det_bbox, res_bbox);
	  
      // Check how much they overlap. Only keep if the overlapped percent isn't too high
	  if((overlap.area() / std::min(det_bbox.area(), res_bbox.area())) > d_->max_overlap)
      {
        should_add = false;
		break;
      }
    }
    if (should_add) // It doesn't overlap too much, add it in
    {
      results->add(*det);
    }
  }

  // We've got our detections, now scale
  results->scale(d_->scale_factor);

  return results;
}

}}} // end namespace kwiver
