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
 * \brief Implementation of ocv::three_frame_differencing
 */

#include <deque>

#include "three_frame_differencing.h"

#include <vital/exceptions.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

using namespace kwiver::vital;


//-----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ------------------------------ Sprokit ------------------------------------


/// Private implementation class
class three_frame_differencing::priv
{
public:
  /// Parameters
  std::size_t m_frame_separation;
  double m_max_foreground_fract;
  double m_max_foreground_fract_thresh;
  std::deque<cv::Mat> m_frames;
  kwiver::vital::logger_handle_t m_logger;

  /// Constructor
  priv()
     :
       m_frame_separation(1),
       m_max_foreground_fract(1),
       m_max_foreground_fract_thresh(-1)
  {
  }

  /// Flush the image queue.
  void reset()
  {
  }

  void
  process_image(cv::Mat &cv_src, cv::Mat &fgmask)
  {
    // Images are in temporal order A (oldest), B, C (newest).
    cv::Mat imgA, imgB, imgC;
    cv_src.copyTo(imgC);

    // Do whatever pre-processing

    m_frames.push_front(imgC);

    if( m_frames.size() < 2*m_frame_separation )
    {
      LOG_TRACE( m_logger, "Haven't collected enough frames yet, so setting "
                           "foreground mask to all zeros.");
      fgmask = cv::Mat(cv_src.rows, cv_src.cols, CV_8UC1, cvScalar(0));
      return;
    }

    LOG_TRACE( m_logger, "Getting frame from end of queue");
    imgA = m_frames.back();
    LOG_TRACE( m_logger, "Getting frame at index frame_separation");
    imgB = m_frames[m_frame_separation];

    if( m_frames.size() > 2*m_frame_separation )
    {
      LOG_TRACE( m_logger, "Removing frame from end of queue");
      m_frames.pop_back();
    }

    // unsigned_sum (default)
    ///  = | | A - C | + | C - B | - | A - B | |
    cv::Mat AminusC, CminusB, AminusB;
    cv::absdiff( imgA, imgC, AminusC );
    cv::absdiff( imgC, imgB, CminusB );
    cv::absdiff( imgA, imgB, AminusB );
    fgmask = cv::abs( AminusC + CminusB - AminusB );

    //cv::cvtColor(fgmask, fgmask, CV_RGB2GRAY, 1);

    if( fgmask.channels() > 1 )
    {
      LOG_TRACE( m_logger, "Converting multichannel foreground mask to single "
                 "channel");
      // Calculate RMS value over channels to create a single channel fgmask.
      cv::Mat fgmask_split[fgmask.channels()];
      cv::split(fgmask, fgmask_split);
      cv::Mat accum = cv::Mat(fgmask.rows, fgmask.cols, CV_32F, cvScalar(0));
      for( int i=0; i<fgmask.channels(); ++i)
      {
        cv::Mat temp;
        fgmask_split[i].convertTo(temp, CV_32F);

        // Divide the result by 3^2 so that the difference has the same scale as
        // a mono image
        cv::multiply( temp, temp, temp, 1/9.0 );
        accum += temp;
      }
      cv::sqrt(accum, accum);
      accum.convertTo(fgmask, CV_8UC1);
    }

    if( true )
    {
      double min_val, max_val;
      cv::minMaxLoc(fgmask, &min_val, &max_val);
      LOG_TRACE( m_logger, "heat map min: " + std::to_string( min_val ) +
                 " max: " + std::to_string( max_val ) );
    }

    if( m_max_foreground_fract < 1)
    {
      int total_pixels = fgmask.rows*fgmask.cols;
      int max_fg_pixels = total_pixels*m_max_foreground_fract;
      cv::Mat mask;
      cv::threshold( fgmask, mask, m_max_foreground_fract_thresh, 1,
                     cv::THRESH_BINARY );
      int nonzero_pixels = cv::sum(mask).val[0];
      LOG_TRACE( m_logger, (double)nonzero_pixels/(double)total_pixels*100 <<
                 "% foreground pixels." );
      if( nonzero_pixels > max_fg_pixels )
      {
        LOG_DEBUG( m_logger, "Foreground pixels exceed maximum set to " <<
                   m_max_foreground_fract*100 << "%, something must have "
                   "failed. Resetting background model." );

        // Reset background model, but wait until next iteration to start
        // updating it because the current frame might be bad.
        reset();
        fgmask = cv::Scalar(0);
      }
    }
  }
};


/// Constructor
three_frame_differencing
::three_frame_differencing()
: d_(new priv)
{
  attach_logger( "arrows.ocv.three_frame_differencing" );
  d_->m_logger = logger();
  d_->reset();
}


/// Destructor
three_frame_differencing
::~three_frame_differencing() VITAL_NOTHROW
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
three_frame_differencing
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "frame_separation", d_->m_frame_separation,
                     "Number of frames of separation for difference "
                     "calculation. Queue of collected images must be twice this"
                     "value before a three-frame difference can be calculated." );
  config->set_value( "max_foreground_fract", d_->m_max_foreground_fract,
                     "Specifies the maximum expected fraction of the scene "
                     "that may contain foreground movers at any time. When the "
                     "fraction of pixels determined to be in motion exceeds "
                     "this value, the background model is assumed to be "
                     "invalid (e.g., due to excessive camera motion) and is "
                     "reset. The default value of 1 indicates that no checking "
                     "is done." );
  config->set_value( "max_foreground_fract_thresh",
                     d_->m_max_foreground_fract_thresh,
                     "To be used in conjunction with max_foreground_fract, this"
                     "parameter defines the threshold for foreground in order "
                     "to determine if the maximum fraction of foreground has "
                     "been exceeded." );

  return config;
}


/// Set this algo's properties via a config block
void
three_frame_differencing
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->m_frame_separation   = config->get_value<int>( "frame_separation" );
  d_->m_max_foreground_fract   = config->get_value<double>( "max_foreground_fract" );
  d_->m_max_foreground_fract_thresh   = config->get_value<double>( "max_foreground_fract_thresh" );

  if( d_->m_frame_separation < 0 )
  {
    throw algorithm_configuration_exception( type_name(), impl_name(),
                                             "frame_separation must be an "
                                             "integer greater than 0." );
  }

  if( d_->m_max_foreground_fract < 0 || d_->m_max_foreground_fract > 1 )
  {
    throw algorithm_configuration_exception( type_name(), impl_name(),
                                             "max_foreground_fract must be in "
                                             "the range 0-1." );
  }

  if( d_->m_max_foreground_fract != 1 && d_->m_max_foreground_fract_thresh < 0 )
  {
    throw algorithm_configuration_exception( type_name(), impl_name(),
                                             "max_foreground_fract_thresh must "
                                             "be set as a positive value." );
  }

  LOG_DEBUG( logger(), "frame_separation: " << std::to_string(d_->m_frame_separation));
  LOG_DEBUG( logger(), "max_foreground_fract: " << std::to_string(d_->m_max_foreground_fract));
  LOG_DEBUG( logger(), "max_foreground_fract_thresh: " << std::to_string(d_->m_max_foreground_fract_thresh));
}


bool
three_frame_differencing
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


/// Return homography to stabilize the image_src relative to the key frame
image_container_sptr
three_frame_differencing
::process_image( const timestamp& ts,
                 const image_container_sptr image,
                 bool reset_model)
{
  if ( !image)
  {
    throw vital::invalid_data("Inputs to ocv::three_frame_differencing are null");
  }

  if( reset_model )
  {
    d_->reset();
  }

  cv::Mat cv_src, fgmask;
  cv_src = ocv::image_container::vital_to_ocv(image->get_image());

  d_->process_image(cv_src, fgmask);

  return std::make_shared<ocv::image_container>(fgmask);;
}


} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
