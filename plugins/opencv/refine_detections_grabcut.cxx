// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of OCV refine detections grabcut algorithm

#include "refine_detections_grabcut.h"
#include "refine_detections_util.h"

#include <algorithm>

#include <vital/vital_config.h>
#include <vital/exceptions/io.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

// ----------------------------------------------------------------------------
// Private implementation class
class refine_detections_grabcut::priv
{
public:
  priv()
  {
  }

  int iter_count = 2;
  double context_scale_factor = 2;
  bool seed_with_existing_masks = true;
  double foreground_scale_factor = 0;
};

// ----------------------------------------------------------------------------
// Constructor
refine_detections_grabcut
::refine_detections_grabcut()
: d_( new priv() )
{
}


// Destructor
refine_detections_grabcut
::~refine_detections_grabcut()
{
}

// ----------------------------------------------------------------------------
// Get this algorithm's vital::config_block configuration block
kv::config_block_sptr
refine_detections_grabcut
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();
  config->set_value( "iter_count", d_->iter_count,
                     "Number of iterations GrabCut should perform "
                     "for each detection" );
  config->set_value( "context_scale_factor", d_->context_scale_factor,
                     "Amount to scale the detection by to produce a context region" );
  config->set_value( "seed_with_existing_masks", d_->seed_with_existing_masks,
                     "If true, use existing masks as \"certainly foreground\""
                     " seed regions" );
  config->set_value( "foreground_scale_factor", d_->foreground_scale_factor,
                     "Amount to scale the detection by to produce a region "
                     "considered certainly foreground" );
  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
refine_detections_grabcut
::set_configuration( kv::config_block_sptr in_config )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->iter_count = config->get_value< int >( "iter_count" );
  d_->context_scale_factor = config->get_value< double >( "context_scale_factor" );
  d_->seed_with_existing_masks =
    config->get_value< bool >( "seed_with_existing_masks" );
  d_->foreground_scale_factor =
    config->get_value< double >( "foreground_scale_factor" );
}

// ----------------------------------------------------------------------------
// Check that the algorithm's current configuration is valid
bool
refine_detections_grabcut
::check_configuration(kv::config_block_sptr config) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Set detection segmentation masks using cv::grabCut
kv::detected_object_set_sptr
refine_detections_grabcut
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  if( !image_data || !detections )
  {
    return detections;
  }

  using ic = ocv::image_container;
  cv::Mat img = ic::vital_to_ocv( image_data->get_image(), ic::BGR_COLOR );
  cv::Rect img_rect( 0, 0, img.cols, img.rows );

  auto result = std::make_shared< kv::detected_object_set >();
  for( auto const& det : *detections )
  {
    auto&& bbox = det->bounding_box();
    // Determine context crop and translate the bounding box
    auto cbbox = kv::scale_about_center( bbox, d_->context_scale_factor );
    auto ctx_rect = bbox_to_mask_rect( cbbox ) & img_rect;
    auto rect = bbox_to_mask_rect( bbox );
    auto mask_rect = ctx_rect | rect;
    // Perform GrabCut
    cv::Mat mask( mask_rect.size(), CV_8UC1, cv::GC_BGD );
    auto mask_roi = [&]( cv::Rect r ){
      return r.empty() ? cv::Mat() : mask( r - mask_rect.tl() );
    };
    cv::Mat bgdModel, fgdModel;
    mask_roi( rect & ctx_rect ) = cv::GC_PR_FGD;
    if( d_->seed_with_existing_masks && det->mask() )
    {
      mask_roi( rect ).setTo( cv::GC_FGD, get_standard_mask( det ) );
    }
    else
    {
      auto fgbbox = kv::scale_about_center( bbox, d_->foreground_scale_factor );
      mask_roi( bbox_to_mask_rect( fgbbox ) & mask_rect ) = cv::GC_FGD;
    }
    // In two cases calling cv::grabCut doesn't make sense, so skip it:
    // - rect is outside of img_rect, so there's no foreground
    // - rect contains ctx_rect, so there's no background
    if( !( rect & img_rect ).empty() && ( rect & ctx_rect ) != ctx_rect )
    {
      cv::Mat crop_mask = mask_roi( ctx_rect );
      cv::grabCut( img( ctx_rect ), crop_mask, cv::Rect(), bgdModel, fgdModel,
                   d_->iter_count, cv::GC_INIT_WITH_MASK );
    }
    // Crop mask and relabel pixels
    mask = mask_roi( rect ).clone();
    std::array< uint8_t, 256 > lut;
    lut[ cv::GC_BGD ] = lut[ cv::GC_PR_BGD ] = 0;
    lut[ cv::GC_FGD ] = lut[ cv::GC_PR_FGD ] = 1;
    cv::LUT( mask, lut, mask );
    // Add detection with mask to the output
    auto new_det = det->clone();
    // mask is a single-channel image, so the ic::ColorMode argument
    // should be irrelevant
    new_det->set_mask( std::make_shared< ic >( mask, ic::OTHER_COLOR ) );
    result->add( std::move( new_det ) );
  }
  return result;
}

} // end namespace viame
