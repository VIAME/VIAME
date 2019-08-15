/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of ocv::detect_motion_MOG2
 */

#include "detect_motion_mog2.h"

#include <vital/exceptions.h>
#include <vital/types/matrix.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

using namespace kwiver::vital;


//-----------------------------------------------------------------------------

/// Private implementation class
class detect_motion_mog2::priv
{
public:
  /// Parameters
  int m_frame_count;
  int m_history;
  double m_var_threshold;
  double m_learning_rate;
  int m_blur_kernel_size;
  int m_min_frames;
  int m_nmixtures;
  double m_max_foreground_fract;
  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_model;
  image_container_sptr motion_heat_map;
  kwiver::vital::logger_handle_t m_logger;

  /// Constructor
  priv()
     :
       m_frame_count(0),
       m_history(100),
       m_var_threshold(36.0),
       m_learning_rate(0.01),
       m_blur_kernel_size(3),
       m_min_frames(1),
       m_nmixtures(3),
       m_max_foreground_fract(1)
  {
  }

  /// Create new impl instance based on current parameters
  void reset()
  {
    m_frame_count = 0;
#ifdef KWIVER_HAS_OPENCV_VER_3
    bg_model = cv::createBackgroundSubtractorMOG2( m_history, m_var_threshold, false );
    bg_model->setNMixtures( m_nmixtures );
#else
    bg_model = new cv::BackgroundSubtractorMOG2(m_history, m_var_threshold, false);
    bg_model->set("nmixtures", m_nmixtures);
#endif
  }
};


/// Constructor
detect_motion_mog2
::detect_motion_mog2()
: d_(new priv)
{
  attach_logger( "arrows.ocv.detect_motion_mog2" );
  d_->m_logger = logger();
  d_->reset();
}


/// Destructor
detect_motion_mog2
::~detect_motion_mog2() noexcept
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_motion_mog2
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "var_threshold", d_->m_var_threshold,
                     "Threshold on the squared Mahalanobis distance between "
                     "the pixel and the model to decide whether a pixel is "
                     "well described by the background model. This parameter "
                     "does not affect the background update." );
  config->set_value( "history", d_->m_history,
                     "Length of the history." );
  config->set_value( "learning_rate", d_->m_learning_rate,
                     "determines how quickly features are “forgotten” from "
                     "histograms (range 0-1)." );
  config->set_value( "blur_kernel_size", d_->m_blur_kernel_size,
                     "Diameter of the normalized box filter blurring "
                     "kernel (positive integer)." );
  config->set_value( "min_frames", d_->m_min_frames,
                     "Minimum frames that need to be included in the "
                     "background model before detections are emmited." );
  config->set_value( "max_foreground_fract", d_->m_max_foreground_fract,
                     "Specifies the maximum expected fraction of the scene "
                     "that may contain foreground movers at any time. When the "
                     "fraction of pixels determined to be in motion exceeds "
                     "this value, the background model is assumed to be "
                     "invalid (e.g., due to excessive camera motion) and is "
                     "reset. The default value of 1 indicates that no checking "
                     "is done." );

  return config;
}


/// Set this algo's properties via a config block
void
detect_motion_mog2
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->m_var_threshold          = config->get_value<double>( "var_threshold" );
  d_->m_history                = config->get_value<int>( "history" );
  d_->m_learning_rate          = config->get_value<double>( "learning_rate" );
  d_->m_blur_kernel_size       = config->get_value<int>( "blur_kernel_size" );
  d_->m_min_frames             = config->get_value<int>( "min_frames" );
  d_->m_max_foreground_fract   = config->get_value<double>( "max_foreground_fract" );
  if( d_->m_max_foreground_fract < 0 || d_->m_max_foreground_fract > 1 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "max_foreground_fract must be in "
                                             "the range 0-1." );
  }

  if( d_->m_min_frames < 0 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "min_frames must be greater than zero." );
  }

  LOG_DEBUG( logger(), "var_threshold: " << std::to_string(d_->m_var_threshold));
  LOG_DEBUG( logger(), "history: " << std::to_string(d_->m_history));
  LOG_DEBUG( logger(), "learning_rate: " << std::to_string(d_->m_learning_rate));
  LOG_DEBUG( logger(), "blur_kernel_size: " << std::to_string(d_->m_blur_kernel_size));
  LOG_DEBUG( logger(), "max_foreground_fract: " << std::to_string(d_->m_max_foreground_fract));
}


bool
detect_motion_mog2
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


/// Detect motion from a sequence of images
image_container_sptr
detect_motion_mog2
::process_image( const timestamp& ts,
                 const image_container_sptr image,
                 bool reset_model)
{
  if ( !image)
  {
    VITAL_THROW( vital::invalid_data, "Inputs to ocv::detect_motion_mog2 are null");
  }

  if( reset_model )
  {
    LOG_TRACE( logger(), "Received command to reset background model");
    d_->reset();
  }

  cv::Mat cv_src;
  ocv::image_container::vital_to_ocv(image->get_image(),image_container::BGR_COLOR).copyTo(cv_src);

  if( d_->m_blur_kernel_size > 0)
  {
    cv::blur(cv_src, cv_src, cv::Size(d_->m_blur_kernel_size, d_->m_blur_kernel_size) );
  }

  cv::Mat fgmask;
#ifdef KWIVER_HAS_OPENCV_VER_3
  d_->bg_model->apply( cv_src, fgmask, d_->m_learning_rate );
#else
  d_->bg_model->operator()(cv_src, fgmask, d_->m_learning_rate);
#endif
  LOG_TRACE( logger(), "Finished MOG2 motion detector for this iteration");

  ++ d_->m_frame_count;

  if( d_->m_frame_count < d_->m_min_frames )
  {
    LOG_TRACE( logger(), "Haven't collected enough frames yet, so setting "
                         "foreground mask to all zeros.");
    fgmask = cv::Scalar(0);
  }
  else
  {
    if( d_->m_max_foreground_fract < 1)
    {
      int total_pixels = fgmask.rows*fgmask.cols;
      int max_fg_pixels = total_pixels*d_->m_max_foreground_fract;
      int nonzero_pixels = cv::countNonZero(fgmask);
      LOG_TRACE( logger(), (double)nonzero_pixels/(double)total_pixels*100 <<
                 "% foreground pixels." );
      if( nonzero_pixels > max_fg_pixels )
      {
        LOG_DEBUG( logger(), "Foreground pixels exceed maximum set to " <<
                   d_->m_max_foreground_fract*100 << "%, something must have "
                   "failed. Resetting background model." );

        // Reset background model, but wait until next iteration to start
        // updating it because the current frame might be bad.
        d_->reset();
        fgmask = cv::Scalar(0);
      }
    }
  }

  d_->motion_heat_map = std::make_shared<ocv::image_container>(fgmask, image_container::BGR_COLOR);

  return d_->motion_heat_map;
}


} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
