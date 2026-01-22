/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "refine_detections_nms.h"

namespace viame {

namespace kv = kwiver::vital;

/// Private implementation class
class refine_detections_nms::priv
{
public:

  /// Constructor
  priv()
  : nms_scale_factor( 1.0 ),
    output_scale_factor( 1.0 ),
    max_scale_difference( 4.0 ),
    min_scale_difference( 0.25 ),
    max_overlap( 0.5 )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  double nms_scale_factor;
  double output_scale_factor;
  double max_scale_difference;
  double min_scale_difference;
  double max_overlap;
};


/// Constructor
refine_detections_nms
::refine_detections_nms()
: d_( new priv() )
{
}


/// Destructor
refine_detections_nms
::~refine_detections_nms()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
kv::config_block_sptr
refine_detections_nms
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();

  config->set_value( "nms_scale_factor", d_->nms_scale_factor,
                     "The factor by which the detections are scaled during NMS." );

  config->set_value( "output_scale_factor", d_->output_scale_factor,
                     "The factor by which the refined final detections are scaled." );

  config->set_value( "max_scale_difference", d_->max_scale_difference,
                     "If the ratio of the areas of two boxes are different by more "
                     "than this amount [1.0,inf], then don't suppress them." );

  config->set_value( "max_overlap", d_->max_overlap,
                     "The maximum percent a detection can overlap with another "
                     "before it's discarded. Range [0.0,1.0]." );

  return config;
}


/// Set this algorithm's properties via a config block
void
refine_detections_nms
::set_configuration( kv::config_block_sptr in_config )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->nms_scale_factor = config->get_value<double>( "nms_scale_factor" );
  d_->output_scale_factor = config->get_value<double>( "output_scale_factor" );
  d_->max_scale_difference = config->get_value<double>( "max_scale_difference" );
  d_->max_overlap = config->get_value<double>( "max_overlap" );

  if( d_->max_scale_difference != 0 )
  {
    if( d_->max_scale_difference < 1.0 )
    {
      d_->min_scale_difference = d_->max_scale_difference;
      d_->max_scale_difference = 1.0 / d_->min_scale_difference;
    }
    else
    {
      d_->min_scale_difference = 1.0 / d_->max_scale_difference;
    }
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
      scale_about_center( det->bounding_box(), d_->nms_scale_factor );

    for( auto result : *results )
    {
      kv::bounding_box_d res_bbox =
        scale_about_center( result->bounding_box(), d_->nms_scale_factor );

      kv::bounding_box_d overlap =
        kv::intersection( det_bbox, res_bbox );

      // Check how much they overlap. Only keep if the overlapped percent isn't too high
      if(overlap.min_x() < overlap.max_x() && overlap.min_y() < overlap.max_y() &&
         (overlap.area() / std::min(det_bbox.area(), res_bbox.area())) > d_->max_overlap)
      {
        if(d_->max_scale_difference == 0 ) // disabled
        {
          should_add = false;
          break;
        }
        else
        {
          double area_ratio = det_bbox.area() / res_bbox.area();

          if(area_ratio >= d_->min_scale_difference && area_ratio <= d_->max_scale_difference)
          {
            should_add = false;
            break;
          }
        }
      }
    }
    if(should_add) // It doesn't overlap too much, add it in
    {
      if(d_->output_scale_factor != 1.0)
      {
        kv::bounding_box_d adj_bbox =
        scale_about_center( det->bounding_box(), d_->output_scale_factor );

        det->set_bounding_box( adj_bbox );
      }
      results->add(det);
    }
  }

  return results;
}

} // end namespace viame
