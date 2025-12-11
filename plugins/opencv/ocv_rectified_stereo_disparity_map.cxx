/*ckwg +29
 * Copyright 2020-2025 by Kitware, Inc.
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

#include "ocv_rectified_stereo_disparity_map.h"
#include "ocv_stereo_calibration.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/types/camera_map.h>
#include <vital/io/camera_io.h>
#include <vital/exceptions.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <arrows/ocv/camera_intrinsics.h>
#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;

namespace viame {

class ocv_rectified_stereo_disparity_map::priv
{
public:
  std::string algorithm{ "BM" };
  int min_disparity{ 0 };
  int num_disparities{ 16 };
  int sad_window_size{ 21 };
  int block_size{ 3 };
  int speckle_window_size{ 50 };
  int speckle_range{ 5 };

  bool m_computed_rectification{ false };
  bool m_use_filtered_disparity{ false };
  bool m_set_disparity_as_alpha_chanel{ false };
  bool m_invert_disparity_alpha_chanel{ false };
  std::string m_cameras_directory;

  // Calibration data loaded via shared utility
  stereo_calibration_result m_calibration;

  // Rectification maps
  cv::Mat m_rectification_map11;
  cv::Mat m_rectification_map12;
  cv::Mat m_rectification_map21;
  cv::Mat m_rectification_map22;

  kv::logger_handle_t m_logger;

  // Shared calibration utility
  stereo_calibration m_calibrator;

  cv::Ptr<cv::StereoMatcher> left_matcher;
  cv::Ptr<cv::StereoMatcher> right_matcher;
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> disparity_filter;

  void load_camera_calibration()
  {
    if( !m_calibrator.load_calibration_opencv( m_cameras_directory, m_calibration ) )
    {
      VITAL_THROW( kv::invalid_data,
        "Failed to load calibration from: " + m_cameras_directory );
    }
  }
};


ocv_rectified_stereo_disparity_map
::ocv_rectified_stereo_disparity_map()
  : d( new priv() )
{
  attach_logger( "viame.opencv.ocv_rectified_stereo_disparity_map" );
  d->m_logger = logger();
  d->m_calibrator.set_logger( d->m_logger );
}


ocv_rectified_stereo_disparity_map
::~ocv_rectified_stereo_disparity_map()
{
}


// ---------------------------------------------------------------------------------------
kv::config_block_sptr
ocv_rectified_stereo_disparity_map
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "algorithm", d->algorithm, "Algorithm: BM or SGBM" );
  config->set_value( "min_disparity", d->min_disparity, "Min Disparity" );
  config->set_value( "num_disparities", d->num_disparities, "Disparity count" );
  config->set_value( "sad_window_size", d->sad_window_size, "SAD window size" );
  config->set_value( "block_size", d->block_size, "Block size" );
  config->set_value( "speckle_window_size", d->speckle_window_size, "Speckle Window Size" );
  config->set_value( "speckle_range", d->speckle_range, "Speckle Range" );
  config->set_value("use_filtered_disparity", d->m_use_filtered_disparity,
                    "(bool) if true, returns WLS filtered disparity. Raw disparity otherwise.");
  config->set_value("set_disparity_as_alpha_chanel", d->m_use_filtered_disparity,
                    "(bool) if true, combines disparity map with left image as alpha chanel.");
  config->set_value("invert_disparity_alpha_chanel", d->m_invert_disparity_alpha_chanel,
                    "(bool) if true, invert disparity map when used as alpha chanel.");

  config->set_value("cameras_directory", d->m_cameras_directory, "Path to a directory to read cameras from.");

  return config;
}

// ---------------------------------------------------------------------------------------
void ocv_rectified_stereo_disparity_map
::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->algorithm = config->get_value< std::string >( "algorithm" );
  d->min_disparity = config->get_value< int >( "min_disparity" );
  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );
  d->block_size = config->get_value< int >( "block_size" );
  d->speckle_window_size = config->get_value< int >( "speckle_window_size" );
  d->speckle_range = config->get_value< int >( "speckle_range" );
  d->m_use_filtered_disparity = config->get_value< bool >("use_filtered_disparity" );
  d->m_set_disparity_as_alpha_chanel = config->get_value< bool >("set_disparity_as_alpha_chanel" );
  d->m_invert_disparity_alpha_chanel = config->get_value< bool >("invert_disparity_alpha_chanel" );

  d->m_computed_rectification = false;
  d->m_cameras_directory = config->get_value< std::string >( "cameras_directory" );

  d->load_camera_calibration();

  if( d->algorithm == "BM" )
  {
    d->left_matcher = cv::StereoBM::create(d->num_disparities, d->sad_window_size );
    d->left_matcher->setSpeckleWindowSize(d->speckle_window_size);
    d->left_matcher->setSpeckleRange (d->speckle_range);
  }
  else if( d->algorithm == "SGBM" )
  {
    int block_size_squared = d->block_size * d->block_size;
    int p1{8 * block_size_squared};
    int p2{32 * block_size_squared};
    int disp12_max_diff{0};
    int prefilter_cap{0};
    int uniqueness_ratio{10};
    d->left_matcher = cv::StereoSGBM::create(d->min_disparity, d->num_disparities, d->block_size, p1, p2, disp12_max_diff,
                                             prefilter_cap, uniqueness_ratio, d->speckle_window_size, d->speckle_range);
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }

  if(d->m_use_filtered_disparity) {
    d->disparity_filter = cv::ximgproc::createDisparityWLSFilter(d->left_matcher);
    d->right_matcher = cv::ximgproc::createRightMatcher(d->left_matcher);
  }else{
    d->disparity_filter.release();
    d->right_matcher.release();
  }
}


// ---------------------------------------------------------------------------------------
bool ocv_rectified_stereo_disparity_map
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// ---------------------------------------------------------------------------------------
kv::image_container_sptr ocv_rectified_stereo_disparity_map
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  if(left_image->get_image().size() != right_image->get_image().size())
  {
    LOG_WARN(d->m_logger, "Inconsistent left/right images size.");
    return kwiver::vital::image_container_sptr();
  }

  // Load cameras and compute needed rectification matrix
  if( !d->m_computed_rectification )
  {
    LOG_DEBUG( d->m_logger, "Compute rectification matrix" );
    cv::Size img_size = cv::Size(
      left_image->get_image().width(), left_image->get_image().height() );

    const auto& cal = d->m_calibration;
    cv::initUndistortRectifyMap(
      cal.left.camera_matrix, cal.left.dist_coeffs, cal.R1, cal.P1,
      img_size, CV_16SC2, d->m_rectification_map11, d->m_rectification_map12 );
    cv::initUndistortRectifyMap(
      cal.right.camera_matrix, cal.right.dist_coeffs, cal.R2, cal.P2,
      img_size, CV_16SC2, d->m_rectification_map21, d->m_rectification_map22 );

    if( !d->m_rectification_map11.empty() ||
        !d->m_rectification_map12.empty() ||
        !d->m_rectification_map21.empty() ||
        !d->m_rectification_map22.empty() )
    {
      d->m_computed_rectification = true;
    }
  }

  // apply rectification then compute depth map
  cv::Mat ocv1 = kwiver::arrows::ocv::image_container::vital_to_ocv( left_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv2 = kwiver::arrows::ocv::image_container::vital_to_ocv( right_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR  );

  // Convert to grayscale using shared utility
  cv::Mat ocv1_gray = stereo_calibration::to_grayscale( ocv1 );
  cv::Mat ocv2_gray = stereo_calibration::to_grayscale( ocv2 );

  cv::Mat img1r, img2r;
  cv::remap(ocv1_gray, img1r, d->m_rectification_map11, d->m_rectification_map12, cv::INTER_LINEAR);
  cv::remap(ocv2_gray, img2r, d->m_rectification_map21, d->m_rectification_map22, cv::INTER_LINEAR);

  // compute disparity map
  cv::Mat left_disparity_map, left_disparity_float;
  d->left_matcher->compute(img1r, img2r, left_disparity_map);
  left_disparity_map.convertTo(left_disparity_float, CV_32F);

  // Filter disparity map
  if (d->m_use_filtered_disparity && d->right_matcher && d->disparity_filter) {
    auto roi = cv::Rect();
    cv::Mat right_disparity_map, left_filtered_disparity;
    d->right_matcher->compute(img2r, img1r, right_disparity_map);
    d->disparity_filter->filter(left_disparity_map, img1r, left_filtered_disparity, right_disparity_map, roi, img2r);
    left_filtered_disparity.convertTo(left_disparity_float, CV_32F);
  }

  // Convert 16 bits fixed-point disparity map (where each disparity value has 4 fractional bits)
  // from  StereoBM or StereoSGBM
  // cf https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html
  left_disparity_float /= 16.0;

  if (d->m_set_disparity_as_alpha_chanel){
    // Convert left image to RGBA
    cv::Mat left_rgba, dest_tmp;

    cv::remap(ocv1, left_rgba, d->m_rectification_map11, d->m_rectification_map12, cv::INTER_LINEAR);
    cv::cvtColor(left_rgba, left_rgba, left_rgba.channels() > 1 ? cv::COLOR_BGR2BGRA :  cv::COLOR_GRAY2BGRA);

    // Convert disparity to 8bit for compatibility with alpha chanel output
    left_disparity_float.convertTo(dest_tmp, CV_8UC1);

    // Invert disparity map and keep out of focus pixels as black if needed
    if(d->m_invert_disparity_alpha_chanel) {
      // Set black pixels to white
      cv::Mat mask;
      inRange(dest_tmp, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), mask);
      dest_tmp.setTo(cv::Scalar(255, 255, 255), mask);

      // Invert mat
      cv::bitwise_not(dest_tmp, dest_tmp);
    }

    // Set inverted disparity map as alpha value of left image
    std::vector<cv::Mat>channels(4);
    cv::split(left_rgba, channels);
    channels[3] = dest_tmp;
    cv::merge(channels, left_disparity_float);
  }

  return kv::image_container_sptr(
      new kwiver::arrows::ocv::image_container(left_disparity_float, kwiver::arrows::ocv::image_container::BGR_COLOR));
}

} //end namespace viame
