/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * \brief Implementation of ocv::track_features_klt
 */

#include "track_features_klt.h"
//#include <arrows/core/merge_tracks.h>
#include <arrows/ocv/image_container.h>

#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

#include <vital/vital_foreach.h>
#include <vital/algo/detect_features.h>

#include <vital/exceptions/algorithm.h>
#include <vital/exceptions/image.h>

#include <opencv2/video/tracking.hpp>

#include <kwiversys/SystemTools.hxx>

using namespace kwiver::vital;
typedef kwiversys::SystemTools ST;

namespace kwiver {
namespace arrows {
namespace ocv {


/// Private implementation class
class track_features_klt::priv
{
public:
  /// Constructor
  priv()
  {
  }

  /// The feature detector algorithm to use
  vital::algo::detect_features_sptr detector;

  cv::Mat prev_image;

  std::vector<cv::Point2f> prev_points;
};


/// Default Constructor
track_features_klt
::track_features_klt()
: d_(new priv)
{
}


/// Destructor
track_features_klt
::~track_features_klt() VITAL_NOTHROW
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
track_features_klt
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm
  algo::detect_features::
    get_nested_algo_configuration("feature_detector", config, d_->detector);

  return config;
}


/// Set this algo's properties via a config block
void
track_features_klt
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::detect_features_sptr df;
  algo::detect_features::set_nested_algo_configuration("feature_detector", config, df);
  d_->detector = df;
}


bool
track_features_klt
::check_configuration(vital::config_block_sptr config) const
{
  return algo::detect_features::check_nested_algo_configuration("feature_detector", config);
}


/// Extend a previous set of tracks using the current frame
track_set_sptr
track_features_klt
::track(track_set_sptr prev_tracks,
        unsigned int frame_number,
        image_container_sptr image_data,
        image_container_sptr mask) const
{
  // verify that all dependent algorithms have been initialized
  if( !d_->detector )
  {
    // Something did not initialize
    throw vital::algorithm_configuration_exception(this->type_name(), this->impl_name(),
        "not all sub-algorithms have been initialized");
  }

  track_set_sptr existing_set;
  feature_set_sptr curr_feat;

  cv::Mat cv_img = ocv::image_container::vital_to_ocv(image_data->get_image());
  cv::Mat cv_mask;

  // Only initialize a mask image if the given mask image container contained
  // valid data.
  if( mask && mask->size() > 0 )
  {
    if ( image_data->width() != mask->width() ||
         image_data->height() != mask->height() )
    {
      throw image_size_mismatch_exception(
          "OCV KLT feature tracker algorithm given a non-zero mask with mismatched "
          "shape compared to input image",
          image_data->width(), image_data->height(),
          mask->width(), mask->height()
          );
    }

    // Make sure we make a one-channel cv::Mat
    vital::image s = mask->get_image();
    // hijacking memory of given mask image, but only telling the new image
    // object to consider the first channel. See vital::image documentation.
    vital::image i(s.memory(),
                   s.first_pixel(),
                   s.width(),  s.height(), 1 /*depth*/,
                   s.w_step(), s.h_step(), s.d_step(), s.pixel_traits());
    cv_mask = ocv::image_container::vital_to_ocv(i);
  }


  track_id_t next_track_id = 0;


  // compute features and descriptors from the image
  if( d_->prev_points.empty() )
  {
    // see if there are already existing tracks on this frame
    if( prev_tracks )
    {
      existing_set = prev_tracks->active_tracks(frame_number);
      if( existing_set && existing_set->size() > 0 )
      {
        LOG_DEBUG( logger(), "Using existing features on frame "<<frame_number);
        // use existing features
        curr_feat = existing_set->frame_features(frame_number);
      }
    }
    if( !curr_feat || curr_feat->size() == 0 )
    {
      LOG_DEBUG( logger(), "Computing new features on frame "<<frame_number);
      // detect features on the current frame
      curr_feat = d_->detector->detect(image_data, mask);
    }
    std::vector<feature_sptr> vf = curr_feat->features();

    typedef std::vector<feature_sptr>::const_iterator feat_itr;
    feat_itr fit = vf.begin();
    std::vector<vital::track_sptr> new_tracks;
    d_->prev_points.clear();
    for(; fit != vf.end(); ++fit)
    {
       track::track_state ts(frame_number, *fit, nullptr);
       new_tracks.push_back(vital::track_sptr(new vital::track(ts)));
       new_tracks.back()->set_id(next_track_id++);
       d_->prev_points.push_back(cv::Point2f((*fit)->loc().x(), (*fit)->loc().y()));
    }
    d_->prev_image = cv_img;
    return track_set_sptr(new simple_track_set(new_tracks));
  }

  std::vector<cv::Point2f> new_points;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(d_->prev_image, cv_img, d_->prev_points, new_points, status, err);
  d_->prev_image = cv_img.clone();

  // get the last track id in the existing set of tracks and increment it
  next_track_id = (*prev_tracks->all_track_ids().crbegin()) + 1;

  track_set_sptr active_set = prev_tracks->active_tracks();
  std::vector<track_sptr> active_tracks = active_set->tracks();
  std::vector<feature_sptr> vf = active_set->last_frame_features()->features();
  std::vector<track_sptr> all_tracks = prev_tracks->tracks();

  std::vector<cv::Point2f> next_points;
  for(unsigned int i=0; i< active_tracks.size(); ++i)
  {
    vector_2f np(new_points[i].x, new_points[i].y);
    if(!status[i] || np.x() < 0 || np.y() < 0 || np.x() > image_data->width() || np.y() > image_data->height())
    {
      continue;
    }
    auto f = std::make_shared<feature_f>(*vf[i]);
    f->set_loc(np);
    track::track_state ts(frame_number, f, nullptr);
    track_sptr t = active_tracks[i];
    t->append(ts);
    next_points.push_back(new_points[i]);
  }
  d_->prev_points = next_points;

  return std::make_shared<simple_track_set>(all_tracks);
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
