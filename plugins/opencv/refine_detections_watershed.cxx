// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of OCV refine detections watershed algorithm

#include "refine_detections_watershed.h"
#include "refine_detections_util.h"

#include <algorithm>

#include <vital/vital_config.h>
#include <vital/exceptions/io.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

// ----------------------------------------------------------------------------
// Check that the algorithm's current configuration is valid
bool
refine_detections_watershed
::check_configuration(kv::config_block_sptr config) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Set detection segmentation masks using cv::watershed
kv::detected_object_set_sptr
refine_detections_watershed
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

  cv::Mat background( img.size(), CV_8UC1, 255 );
  // Explicitly convert 0 to a Scalar to avoid interpretation as NULL
  cv::Mat markers( img.size(), CV_32SC1, cv::Scalar( 0 ) );

  std::vector< cv::Mat > seeds;
  size_t i;
  for( i = 0; i < detections->size(); ++i )
  {
    auto&& det = detections->at( i );
    auto&& bbox = det->bounding_box();
    auto rect = bbox_to_mask_rect( bbox );
    auto uncertain_bbox = kv::scale_about_center(
      bbox, c_uncertain_scale_factor );
    auto uncertain_rect = bbox_to_mask_rect( uncertain_bbox );
    background( uncertain_rect & img_rect ) = 0;
    auto crop_rect = rect & img_rect;
    cv::Mat m = markers( crop_rect );
    cv::Mat already_set = m != 0;
    cv::Mat seed;
    if( c_seed_with_existing_masks && det->mask() )
    {
      // Clone because this is modified below (crop_mask.setTo)
      seed = get_standard_mask( det ).clone();
    }
    else
    {
      auto seed_bbox = kv::scale_about_center( bbox, c_seed_scale_factor );
      seed = cv::Mat( rect.size(), CV_8UC1, cv::Scalar( 0 ) );
      seed( ( bbox_to_mask_rect( seed_bbox ) & rect ) - rect.tl() ) = 1;
    }
    cv::Mat crop_seed = crop_rect.empty() ? cv::Mat()
      : seed( crop_rect - rect.tl() );
    m.setTo( i + 1, crop_seed );
    m.setTo( -1, crop_seed & already_set );
    seeds.push_back( std::move( seed ) );
  }
  markers = cv::max( markers, 0 );
  markers.setTo( i + 1, background );
  cv::watershed( img, markers );

  auto result = std::make_shared< kv::detected_object_set >();
  for( i = 0; i < detections->size(); ++i )
  {
    auto&& det = detections->at( i );
    auto&& bbox = det->bounding_box();
    auto rect = bbox_to_mask_rect( bbox );
    auto crop_rect = rect & img_rect;
    auto& mask = seeds[ i ];
    cv::Mat crop_mask = crop_rect.empty() ? cv::Mat()
      : mask( crop_rect - rect.tl() );
    crop_mask.setTo( 1, markers( crop_rect ) == i + 1 );
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
