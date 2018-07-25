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

#include <vital/algo/detect_features.h>
#include <vital/logger/logger.h>

#include <vital/exceptions/algorithm.h>
#include <vital/exceptions/image.h>

#include <opencv2/video/tracking.hpp>

#include <arrows/core/track_set_impl.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {


/// Private implementation class
class track_features_klt::priv
{
public:
  /// Constructor
  priv():
    last_detect_num_features(0),
    redetect_threshold(0.7),
    exclusionary_radius_image_frac(0.01),
    win_size(41),
    half_win_size(win_size/2),
    tracked_feat_mask_downsample_fact(1),
    exclude_rad_pixels(1),
    erp2(1),
    max_pyramid_level(3)
  {
  }

  // sets up a mask based on the points.  We query this mask to find out if a
  // newly detected point is near an existing point.
  void set_tracked_feature_location_mask(const std::vector<cv::Point2f> &points,
    image_container_sptr image_data)
  {
    exclude_rad_pixels =
      std::max<int>(1, exclusionary_radius_image_frac *
        std::min(image_data->width(), image_data->height()));

     int log_tracked_feat_mask_downsample_fact = int(floor(log2f(float(exclude_rad_pixels)/2.0f)));

     tracked_feat_mask_downsample_fact = 1 << log_tracked_feat_mask_downsample_fact;

     int tfh = int(ceil(float(image_data->height()) / float(tracked_feat_mask_downsample_fact)));
     int tfw = int(ceil(float(image_data->width()) / float(tracked_feat_mask_downsample_fact)));

    if (tracked_feature_location_mask.rows != tfh ||
        tracked_feature_location_mask.cols != tfw )
    {
      tracked_feature_location_mask = cv::Mat(tfh,tfw, CV_8UC1);
    }

    erp2 = exclude_rad_pixels * exclude_rad_pixels;
    erp2 /= tracked_feat_mask_downsample_fact;
    erp2 /= tracked_feat_mask_downsample_fact;

    //mark the whole tracked feature mask as not having any features
    tracked_feature_location_mask.setTo(0);

    for (auto const& np : points)
    {
      set_exclude_mask(np);
    }
  }

  void set_exclude_mask(cv::Point2f const &pt)
  {
    for (int r = -exclude_rad_pixels; r <= exclude_rad_pixels; ++r)
    {
      int row = r + pt.y / tracked_feat_mask_downsample_fact;
      if (row < 0 || row >= tracked_feature_location_mask.rows)
      {
        continue;
      }
      for (int c = -exclude_rad_pixels; c <= exclude_rad_pixels; ++c)
      {
        int col = c + pt.x / tracked_feat_mask_downsample_fact;
        if (col < 0 || col >= tracked_feature_location_mask.cols)
        {
          continue; // outside of mask image
        }
        if ((r*r + c*c) > erp2)
        {
          continue;  //outside of mask radius
        }
        //set the mask to 1 here
        tracked_feature_location_mask.at<unsigned char>(row, col) = 1;
      }
    }
  }

  bool exclude_mask_is_set(vital::vector_2d const &pt) const {
    return tracked_feature_location_mask.at<unsigned char>(pt.y() / tracked_feat_mask_downsample_fact, pt.x() / tracked_feat_mask_downsample_fact) != 0;
  }

  /// Set current parameter values to the given config block
  void update_config(vital::config_block_sptr &config) const
  {
    config->set_value("redetect_frac_lost_threshold", redetect_threshold,
                      "redetect if fraction of features tracked from last "
                      "detection drops below this level");
    int grid_rows, grid_cols;

    dist_image.get_grid_size(grid_rows, grid_cols);

    config->set_value("grid_rows", grid_rows,
                      "rows in feature distribution enforcing grid");

    config->set_value("grid_cols", grid_cols,
                      "colums in feature distribution enforcing grid");

    config->set_value("new_feat_exclusionary_radius_image_fraction",
                      exclusionary_radius_image_frac,
                      "do not place new features any closer than this fraction of image min "
                      "dimension to existing features");

    config->set_value("win_size", win_size,
                      "klt image patch side length (it's a square)");

    config->set_value("max_pyramid_level", max_pyramid_level,
                      "maximum pyramid level used in klt feature tracking");
  }

  /// Set our parameters based on the given config block
  void set_config(const vital::config_block_sptr & config)
  {
    redetect_threshold =
      config->get_value<double>("redetect_frac_lost_threshold");

    int grid_rows = config->get_value<int>("grid_rows");

    int gridCols = config->get_value<int>("grid_cols");

    dist_image.set_grid_size(grid_rows, gridCols);

    last_detect_distImage.set_grid_size(grid_rows, gridCols);

    exclusionary_radius_image_frac =
      config->get_value<float>("new_feat_exclusionary_radius_image_fraction");

    win_size = config->get_value<int>("win_size");

    half_win_size = win_size / 2;

    max_pyramid_level = config->get_value<int>("max_pyramid_level");
  }

  bool check_configuration(vital::config_block_sptr config) const
  {
    bool success(true);

    float test_redetect_threshold =
      config->get_value<double>("redetect_frac_lost_threshold");

    int test_grid_rows = config->get_value<int>("grid_rows");

    int test_grid_cols = config->get_value<int>("grid_cols");

    float test_exclusionary_radius_image_frac =
      config->get_value<float>("new_feat_exclusionary_radius_image_fraction");

    int test_win_size = config->get_value<int>("win_size");

    if (!(0 < test_redetect_threshold && test_redetect_threshold <= 1.0))
    {
      LOG_ERROR(m_logger, "redetect_frac_lost_threshold ("
        << test_redetect_threshold
        << ") should be greater than zero and <= 1.0");
      success = false;
    }

    if (test_grid_rows <= 0)
    {
      LOG_ERROR(m_logger, "grid_rows (" << test_grid_rows <<
        ") must be greater than 0");
      success = false;
    }

    if (test_grid_cols <= 0)
    {
      LOG_ERROR(m_logger, "grid_cols (" << test_grid_cols <<
        ") must be greater than 0");
      success = false;
    }

    if (!(0 < test_exclusionary_radius_image_frac &&
        test_exclusionary_radius_image_frac < 1.0))
    {
      LOG_ERROR(m_logger, "new_feat_exclusionary_radius_image_fraction ("
        << test_exclusionary_radius_image_frac <<
        ") must be between 0.0 and 1.0");
      success = false;
    }

    if (test_win_size < 3)
    {
      LOG_ERROR(m_logger, "win_size (" << test_win_size <<
        ") must be three or more");
      success = false;
    }

    if (max_pyramid_level < 0)
    {
      LOG_ERROR(m_logger, "max_pyramid_level (" <<
        max_pyramid_level << ") must be non-negative");
      success = false;
    }

    return success;
  }

  class feature_distribution_image
  {
  public:
    feature_distribution_image():
      bad_bins_frac_to_redetect(0.125),
      rows(0), cols(0)
    {
      set_grid_size(4, 4);
    }

    void set_grid_size(int _rows, int _cols)
    {
      if (rows != _rows || cols != _cols) {
        rows = _rows;
        cols = _cols;
        dist_image = cv::Mat(rows, cols, CV_16UC1);
        dist_image.setTo(0);
      }
    }
    void get_grid_size(int &_rows, int& _cols) const
    {
      _rows = rows;
      _cols = cols;
    }

    feature_distribution_image& operator=(
      const feature_distribution_image &other)
    {
      if (&other != this)
      {
        other.dist_image.copyTo(dist_image);
      }
        return *this;
    }

    bool should_redetect(const feature_distribution_image &lastDetectDist,
                         float redetect_threshold)
    {
      int bad_bins = 0;
      for (int r = 0; r < rows; ++r)
      {
        for (int c = 0; c < cols; ++c)
        {
          if (dist_image.at<uint16_t>(r, c) <
              uint16_t(redetect_threshold*
              float(lastDetectDist.dist_image.at<uint16_t>(r, c))))
          {
            ++bad_bins;
          }
        }
      }
      if (bad_bins >= int(float(rows*cols)*bad_bins_frac_to_redetect))
      {
        return true;
      }
      return false;
    }

    void set_from_feature_vector(const std::vector<cv::Point2f> &points,
                                 image_container_sptr image_data)
    {
      dist_image.setTo(0);
      const int dist_bin_x_len = int(image_data->width() / cols);
      const int dist_bin_y_len = int(image_data->height() / rows);

      for (auto v = points.begin(); v != points.end(); ++v)
      {
        const cv::Point2f &tp = *v;
        int dist_bin_x =
          std::min<int>(std::max<int>(tp.x / dist_bin_x_len, 0), cols - 1);
        int dist_bin_y =
          std::min<int>(std::max<int>(tp.y / dist_bin_y_len, 0), rows - 1);

        uint16_t& numFeatInBin =
          dist_image.at<uint16_t>(dist_bin_y, dist_bin_x);
        if (numFeatInBin < UINT16_MAX)
        {
          ++numFeatInBin;  //make sure we don't roll over the UINT_16.
        }
      }
    }

    cv::Mat dist_image;
    float bad_bins_frac_to_redetect;
  private:
      int rows, cols;
  };

  /// The feature detector algorithm to use
  vital::algo::detect_features_sptr detector;
  cv::Mat prev_image;
  size_t last_detect_num_features;
  float redetect_threshold;
  cv::Mat tracked_feature_location_mask;
  float exclusionary_radius_image_frac;
  kwiver::vital::logger_handle_t m_logger;
  feature_distribution_image dist_image;
  feature_distribution_image last_detect_distImage;
  int win_size;
  int half_win_size;
  int tracked_feat_mask_downsample_fact;
  int exclude_rad_pixels;
  int erp2;
  int max_pyramid_level;
};


/// Default Constructor
track_features_klt
::track_features_klt()
: d_(new priv)
{
  attach_logger("arrows.ocv.track_features_klt");
  d_->m_logger = this->logger();
}


/// Destructor
track_features_klt
::~track_features_klt() noexcept
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

  d_->update_config(config);

  return config;
}


/// Set this algo's properties via a config block
void
track_features_klt
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are
  // present.  An alternative is to check for key presence before performing a
  // get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::detect_features_sptr df;
  algo::detect_features::set_nested_algo_configuration(
    "feature_detector", config, df);
  d_->detector = df;
  d_->set_config(config);
}


bool
track_features_klt
::check_configuration(vital::config_block_sptr config) const
{
  bool success(true);
  success = algo::detect_features::check_nested_algo_configuration(
    "feature_detector", config) && success;
  success = d_->check_configuration(config) && success;
  return success;
}


/// Extend a previous set of feature tracks using the current frame
feature_track_set_sptr
track_features_klt
::track(feature_track_set_sptr prev_tracks,
        unsigned int frame_number,
        image_container_sptr image_data,
        image_container_sptr mask) const
{
  // verify that all dependent algorithms have been initialized
  if( !d_->detector )
  {
    // Something did not initialize
    throw vital::algorithm_configuration_exception(
      this->type_name(), this->impl_name(),
        "not all sub-algorithms have been initialized");
  }


  cv::Mat cv_img = ocv::image_container::vital_to_ocv(image_data->get_image(), ocv::image_container::RGB_COLOR);
  cv::Mat cv_mask;

  // Only initialize a mask image if the given mask image container contained
  // valid data.
  if( mask && mask->size() > 0 )
  {
    if ( image_data->width() != mask->width() ||
         image_data->height() != mask->height() )
    {
      throw image_size_mismatch_exception(
          "OCV KLT feature tracker algorithm given a non-zero mask with "
          "mismatched shape compared to input image",
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
    cv_mask = ocv::image_container::vital_to_ocv(i, ocv::image_container::OTHER_COLOR);
  }

  //setup stuff complete
  feature_track_set_sptr cur_tracks = prev_tracks;  //no clone here.  It is done in the process.

  //points to be tracked in the next frame.  Empty at first.
  std::vector<cv::Point2f> next_points;

  //gets the active tracks for the previous frame
  std::vector<track_sptr> active_tracks;
  if (cur_tracks)
  {
    active_tracks = cur_tracks->active_tracks();
  }

  //track features if there are any to track
  if (!active_tracks.empty())
  {
    //track
    std::vector<cv::Point2f> tracked_points, prev_points;
    std::vector<uchar> status;
    std::vector<float> err;

    std::vector<track_sptr> prev_klt_tracks;
    for (auto at : active_tracks)
    {
      auto  bk = std::dynamic_pointer_cast<feature_track_state>(at->back());
      if (bk->frame() != (frame_number - 1))
      {
        continue;
      }
      if (bk->descriptor)
      {
        //skip features with descriptors
        continue;
      }
      prev_points.push_back(cv::Point2f(bk->feature->loc().x(), bk->feature->loc().y()));
      prev_klt_tracks.push_back(at);
    }

    cv::calcOpticalFlowPyrLK(d_->prev_image, cv_img, prev_points,
      tracked_points, status, err,cv::Size(d_->win_size, d_->win_size),d_->max_pyramid_level);
    //copy last frame's features
    for (unsigned int kf_feat_i = 0; kf_feat_i< prev_klt_tracks.size(); ++kf_feat_i)
    {
      //first we check if the active track has a descriptor.  If it does, it's not a klt track so we skip over it.
      track_sptr t = prev_klt_tracks[kf_feat_i];

      vector_2f tp(tracked_points[kf_feat_i].x, tracked_points[kf_feat_i].y);
      if (!status[kf_feat_i]
          || tp.x() <= d_->half_win_size
          || tp.y() <= d_->half_win_size
          || tp.x() >= image_data->width()- d_->half_win_size
          || tp.y() >= image_data->height()- d_->half_win_size)
      {
        // skip features that tracked to outside of the image (or the border)
        // or didn't track properly
        continue;
      }
      //info from feature detector (location, scale etc.)
      auto last_fts = std::dynamic_pointer_cast<feature_track_state>(t->back());
      auto f = std::make_shared<feature_f>(*last_fts->feature);
      f->set_loc(tp);  //feature
      //feature, descriptor and frame number together
      auto fts = std::make_shared<feature_track_state>(frame_number);
      fts->feature = f;
      // append the feature's current location to it's track.  Track was picked
      // up with active_tracks() call on previous_tracks.
      t->append(fts);
      next_points.push_back(tracked_points[kf_feat_i]);
      //increment the feature distribution bins
    }
  }

  //did we track enough features from the previous frame?
  bool detect_new_features =
    next_points.size() <=
    size_t(d_->redetect_threshold*double(d_->last_detect_num_features));

  //set the feature distribution image
  d_->dist_image.set_from_feature_vector(next_points, image_data);

  if (!detect_new_features)
  {
    //now check the distribution of features in the image
    if(d_->dist_image.should_redetect(d_->last_detect_distImage,d_->redetect_threshold))
    {
      //this will never be called on the first image so it will work.
      LOG_DEBUG(logger(), "detecting new feature because of distribution");
      detect_new_features = true;
    }
  }

  if( detect_new_features)
  {
    // detect new features
    // see if there are already existing tracks on this frame
    feature_set_sptr detected_feat;
    if(!cur_tracks)
    {
      typedef std::unique_ptr<track_set_implementation> tsi_uptr;
      cur_tracks = std::make_shared<feature_track_set>(
        tsi_uptr(new kwiver::arrows::core::frame_index_track_set_impl()));
    }

    // get the last track id in the existing set of tracks and increment it
    track_id_t next_track_id = 0;
    if (!cur_tracks->all_track_ids().empty())
    {
      next_track_id = (*cur_tracks->all_track_ids().crbegin()) + 1;
    }

    LOG_DEBUG( logger(), "Computing new features on frame "<<frame_number);
    // detect features on the current frame
    detected_feat = d_->detector->detect(image_data, mask);

    // merge new features into existing features (ignore new features near
    // existing features)
    std::vector<feature_sptr> vf = detected_feat->features();

    // make a mask of current image feature positions. This maks keeps
    // features from being kept that are detected near existing tracks.
    d_->set_tracked_feature_location_mask(next_points, image_data);

    typedef std::vector<feature_sptr>::const_iterator feat_itr;
    for(feat_itr fit = vf.begin(); fit != vf.end(); ++fit)
    {
      if (d_->exclude_mask_is_set((*fit)->loc()))
        continue;

      if ((*fit)->loc().x() < d_->win_size ||
          (*fit)->loc().y() < d_->win_size ||
          (*fit)->loc().x() > image_data->width() - d_->win_size ||
          (*fit)->loc().y() > image_data->height() - d_->win_size)
      {
        continue;  //mask out features at the edge of the image.
      }

      auto fts = std::make_shared<feature_track_state>(frame_number);
      fts->feature = *fit;
      auto t = vital::track::create();  //make a new track
      t->append(fts);  //put the feature in this new track
      t->set_id(next_track_id++); //set the new track id
      cur_tracks->insert(t);
      cv::Point2f new_pt = cv::Point2f((*fit)->loc().x(), (*fit)->loc().y());
      next_points.push_back(new_pt);

      d_->set_exclude_mask(new_pt);      //this makes the points earlier in the
      // detection list more likely to be in tracked feature set.  Should sort
      //by strength to get the best features with the highest likelihood.
    }
    //this includes any features tracked to this frame and the new points
    d_->last_detect_num_features = next_points.size();

    //store the last detected feature distribution
    d_->last_detect_distImage.set_from_feature_vector(next_points, image_data);
  }

  //set up previous data structures for next call
  d_->prev_image = cv_img.clone();

  return cur_tracks;
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
