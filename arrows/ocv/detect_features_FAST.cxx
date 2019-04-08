/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * \brief OCV FAST feature detector wrapper implementation
 */

#include "detect_features_FAST.h"

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {


class detect_features_FAST::priv
{
public:
  /// Constructor
  priv()
    : threshold(10),
      nonmaxSuppression(true),
      targetNumDetections(2500)
  {
#ifdef KWIVER_HAS_OPENCV_VER_3
    neighborhood_type = cv::FastFeatureDetector::TYPE_9_16;
#endif
  }

  /// Create a new FAST detector instance with the current parameter values
  cv::Ptr<cv::FastFeatureDetector> create() const
  {
#ifndef KWIVER_HAS_OPENCV_VER_3
    // 2.4.x version constructor
    return cv::Ptr<cv::FastFeatureDetector>(
        new cv::FastFeatureDetector(threshold, nonmaxSuppression)
    );
#else
    // 3.x version constructor
    return cv::FastFeatureDetector::create(threshold, nonmaxSuppression,
                                           neighborhood_type);
#endif
  }

  /// Update the parameters of the given detector with the currently set values
  void update(cv::Ptr<cv::FeatureDetector> detector) const
  {
#ifndef KWIVER_HAS_OPENCV_VER_3
    detector->set("threshold", threshold);
    detector->set("nonmaxSuppression", nonmaxSuppression);
#else
    auto det = detector.dynamicCast<cv::FastFeatureDetector>();
    det->setThreshold(threshold);
    det->setNonmaxSuppression(nonmaxSuppression);
    det->setType(neighborhood_type);
#endif
  }

  /// Update given config block with currently set parameter values
  void update_config( config_block_sptr config ) const
  {
    config->set_value( "threshold", threshold,
                       "Integer threshold on difference between intensity of "
                       "the central pixel and pixels of a circle around this "
                       "pixel" );
    config->set_value( "nonmaxSuppression", nonmaxSuppression,
                       "if true, non-maximum suppression is applied to "
                       "detected corners (keypoints)" );

#ifdef KWIVER_HAS_OPENCV_VER_3
    std::stringstream ss;
    ss << "one of the three neighborhoods as defined in the paper: "
      "TYPE_5_8=" << cv::FastFeatureDetector::TYPE_5_8 << ", "
      "TYPE_7_12=" << cv::FastFeatureDetector::TYPE_7_12 << ", "
      "TYPE_9_16=" << cv::FastFeatureDetector::TYPE_9_16 << ".";
    config->set_value( "neighborhood_type", neighborhood_type , ss.str());
#endif

    config->set_value("target_num_features_detected", targetNumDetections,
                      "algorithm tries to output approximately this many features. "
                      "Disable by setting to negative value.");

  }

  /// Set parameter values based on given config block
  void set_config( config_block_sptr const &config )
  {
    threshold = config->get_value<int>( "threshold" );
    nonmaxSuppression = config->get_value<bool>( "nonmaxSuppression" );

#ifdef KWIVER_HAS_OPENCV_VER_3
    neighborhood_type = config->get_value<int>( "neighborhood_type" );
#endif
    targetNumDetections = config->get_value<int>("target_num_features_detected");
  }

  /// Check config parameter values
  bool check_config(vital::config_block_sptr const &config,
                    logger_handle_t const &logger) const
  {
    bool valid = true;

#ifdef KWIVER_HAS_OPENCV_VER_3
    // Check that the input integer is one of the valid enum values
    int nt = config->get_value<int>( "neighborhood_type" );
    if( ! ( nt == cv::FastFeatureDetector::TYPE_5_8 ||
            nt == cv::FastFeatureDetector::TYPE_7_12 ||
            nt == cv::FastFeatureDetector::TYPE_9_16) )
    {
      std::stringstream ss;
      ss << "FAST feature detector neighborhood type is not one of the valid "
        "values (see config comment). Given " << nt;
      LOG_ERROR( logger, ss.str() );
      valid = false;
    }
#endif

    return valid;
  }

  mutable int threshold;
  bool nonmaxSuppression;
#ifdef KWIVER_HAS_OPENCV_VER_3
  int neighborhood_type;
#endif
  int targetNumDetections;
};


/// Constructor
detect_features_FAST
::detect_features_FAST()
  : p_(new priv)
{
  attach_logger("arrows.ocv.detect_features_FAST");
  detector = p_->create();
}


/// Destructor
detect_features_FAST
::~detect_features_FAST()
{
}


vital::config_block_sptr
detect_features_FAST
::get_configuration() const
{
  vital::config_block_sptr config = ocv::detect_features::get_configuration();
  p_->update_config( config );
  return config;
}


void
detect_features_FAST
::set_configuration(vital::config_block_sptr in_config)
{
  vital::config_block_sptr config = get_configuration();
  config->merge_config( in_config );
  p_->set_config( config );

  // Update the wrapped algo inst with new parameters
  p_->update(detector);
}


bool
detect_features_FAST
::check_configuration(vital::config_block_sptr in_config) const
{
  vital::config_block_sptr config = get_configuration();
  config->merge_config(in_config);
  return p_->check_config( config, logger() );
}

/// Extract a set of image features from the provided image
vital::feature_set_sptr
detect_features_FAST
::detect(vital::image_container_sptr image_data, vital::image_container_sptr mask) const
{
  float close_detect_thresh = 0.1; //be within 10% of target
  const int duplicate_feat_count_thresh = 4;
  auto last_det_feat_set = ocv::detect_features::detect(image_data, mask);

  if (p_->targetNumDetections <= 0)
  {
    return last_det_feat_set;
  }

  if (last_det_feat_set->size() > (1.0 + close_detect_thresh)*p_->targetNumDetections)
  {
    //we got too many features
    int same_num_feat_count = 0;
    while (true)
    {
      auto last_threshold = p_->threshold;
      //make the threshold higher so we detect fewer features
      p_->threshold *= (1.0 + close_detect_thresh);
      if (p_->threshold == last_threshold)
      {
        p_->threshold += 1;
      }
      p_->update(detector);
      // Update the wrapped algo inst with new parameters

      auto higher_thresh_feat_set = ocv::detect_features::detect(image_data, mask);

      if (last_det_feat_set->size() == higher_thresh_feat_set->size())
      {
        ++same_num_feat_count;
      }
      else
      {
        same_num_feat_count = 0;
      }

      if (higher_thresh_feat_set->size() <= static_cast<size_t>(p_->targetNumDetections) ||
          same_num_feat_count > duplicate_feat_count_thresh)
      {
        //ok, we've crossed from too many to too few features
        // or we aren't changing the number of detected features much
        int higher_diff = abs(p_->targetNumDetections - int(higher_thresh_feat_set->size()));
        int last_diff =  abs(int(last_det_feat_set->size()) - p_->targetNumDetections);

        if (higher_diff < last_diff)
        {
          last_det_feat_set = higher_thresh_feat_set;
          // keep existing threshold. it worked.
        }
        else
        {
          // set threshold back to one used in last detection
          p_->threshold = last_threshold;
          p_->update(detector);
        }
        break;
      }
      last_det_feat_set = higher_thresh_feat_set;
      last_threshold = p_->threshold;
    }
  }
  else
  {
    if (last_det_feat_set->size() < (1.0 - close_detect_thresh) * p_->targetNumDetections)
    {
      //we got too few features
      // or we aren't changing the number of detected features much
      int same_num_feat_count = 0;
      while (true)
      {
        auto last_threshold = p_->threshold;
        p_->threshold *= (1.0 - close_detect_thresh);  //make the threshold higer so we detect fewer features
        if (p_->threshold == last_threshold)
        {
          p_->threshold -= 1;
        }
        if (p_->threshold <= 0)
        {
          //can't have a non-positive detection threshold.
          break;
        }
        p_->update(detector);
        auto lower_thresh_feat_set = ocv::detect_features::detect(image_data, mask);

        if (last_det_feat_set->size() == lower_thresh_feat_set->size())
        {
          ++same_num_feat_count;
        }
        else
        {
          same_num_feat_count = 0;
        }

        if (lower_thresh_feat_set->size() >= static_cast<size_t>(p_->targetNumDetections) ||
          same_num_feat_count > duplicate_feat_count_thresh)
        {
          int lower_diff = abs(int(lower_thresh_feat_set->size()) - p_->targetNumDetections);
          int last_diff = abs(int(last_det_feat_set->size()) - p_->targetNumDetections);

          //ok, we've crossed from too few to too many features
          if (lower_diff < last_diff)
          {
            last_det_feat_set = lower_thresh_feat_set;
            // keep existing threshold. it worked.
          }
          else
          {
            // set threshold back to one used in last detection
            p_->threshold = last_threshold;
            p_->update(detector);
          }
          break;
        }
        last_det_feat_set = lower_thresh_feat_set;
        last_threshold = p_->threshold;
      }
    }
  }


  return last_det_feat_set;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
