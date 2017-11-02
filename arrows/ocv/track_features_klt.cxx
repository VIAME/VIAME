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
  priv():
    lastDetectNumFeatures(0),
    redetect_threshold(0.7),
    exclusionary_radius_image_frac(0.01),
    win_size(41),
    half_win_size(win_size/2)
  {
  }

  /// The feature detector algorithm to use
  vital::algo::detect_features_sptr detector;

  cv::Mat prev_image;

  std::vector<cv::Point2f> prev_points;

  size_t lastDetectNumFeatures;
  float redetect_threshold;
  cv::Mat trackedFeatureLocationMask;
  float exclusionary_radius_image_frac;

  //sets up a mask based on the points.  We querry this mask to find out if a newly detected point is near an 
  //existing point.
  void setTrackedFeatureLocationMask(const std::vector<cv::Point2f> &points, image_container_sptr image_data) {
    if (trackedFeatureLocationMask.rows != image_data->height() ||
        trackedFeatureLocationMask.cols != image_data->width())
    {
      trackedFeatureLocationMask = cv::Mat(int(image_data->height()), 
        int(image_data->width()), CV_8UC1); // really this is a bool array so I could pack it more.
    }

    //mark the whole tracked feature mask as not having any features
    int exclusionary_radius_pixels =
      std::max<int>(1, exclusionary_radius_image_frac * std::min(image_data->width(), image_data->height()));

    trackedFeatureLocationMask.setTo(0);
    for (unsigned int i = 0; i < points.size(); ++i) {
      const cv::Point2f &np = points[i];
      for (int r = -exclusionary_radius_pixels; r <= exclusionary_radius_pixels; ++r) {
        for (int c = -exclusionary_radius_pixels; c <= exclusionary_radius_pixels; ++c) {
          if ((r*r + c*c) > exclusionary_radius_pixels * exclusionary_radius_pixels) {
            continue;  //outside of mask radius
          }
          int row = r + np.y;
          int col = c + np.x;
          if (row < 0 || row >= trackedFeatureLocationMask.rows ||
            col < 0 || col >= trackedFeatureLocationMask.cols) {
            continue; // outside of mask image
          }
          trackedFeatureLocationMask.at<unsigned char>(row, col) = 1;  //set the mask to 1 here
        }
      }
    }
  }
  
  /// Set current parameter values to the given config block
  void update_config(vital::config_block_sptr &config) const
  {
    config->set_value("redetect_frac_lost_threshold", redetect_threshold,
                      "redetect if fraction of features tracked from last detection drops below this level");
    int gridRows, gridCols;
    distImage.getGridSize(gridRows, gridCols);
    config->set_value("grid_rows", gridRows, "rows in feature distribution enforcing grid");
    config->set_value("grid_cols", gridCols, "colums in feature distribution enforcing grid");
    config->set_value("new_feat_exclusionary_radius_image_fraction", exclusionary_radius_image_frac,
      "do not place new features any closer than this fraction of image min dimension to existing features");
    config->set_value("win_size", win_size, "klt image patch side length (it's a square)");
  }

  /// Set our parameters based on the given config block
  void set_config(const vital::config_block_sptr & config) {
    redetect_threshold = config->get_value<double>("redetect_frac_lost_threshold");
    distImage.redetect_threshold = redetect_threshold;
    lastDetect_distImage.redetect_threshold = redetect_threshold;
    int gridRows = config->get_value<int>("grid_rows");
    int gridCols = config->get_value<int>("grid_cols");
    distImage.setGridSize(gridRows, gridCols);
    lastDetect_distImage.setGridSize(gridRows, gridCols);
    exclusionary_radius_image_frac = config->get_value<float>("new_feat_exclusionary_radius_image_fraction");
    win_size = config->get_value<int>("win_size");
    half_win_size = win_size / 2;
  }

  bool check_configuration(vital::config_block_sptr config) const {
    bool success(true);

    float test_redetect_threshold = config->get_value<double>("redetect_frac_lost_threshold");
    int test_gridRows = config->get_value<int>("grid_rows");
    int test_gridCols = config->get_value<int>("grid_cols");
    float test_exclusionary_radius_image_frac = 
      config->get_value<float>("new_feat_exclusionary_radius_image_fraction");
    int test_win_size = config->get_value<int>("win_size");

    if (!(0 < test_redetect_threshold && test_redetect_threshold <= 1.0))
    {
      std::stringstream str;
      config->print(str);
      LOG_ERROR(m_logger, "redetect_frac_lost_threshold (" << test_redetect_threshold 
        << ") should be greater than zero and <= 1.0"
        " Configuration is as follows:\n" << str.str());
      success = false;
    }

    if (test_gridRows <= 0)
    {
      std::stringstream str;
      config->print(str);
      LOG_ERROR(m_logger, "grid_rows (" << test_gridRows <<") must be greater than 0"
        " Configuration is as follows:\n" << str.str());
      success = false;
    }

    if (test_gridCols <= 0)
    {
      std::stringstream str;
      config->print(str);
      LOG_ERROR(m_logger, "grid_cols (" << test_gridCols << ") must be greater than 0"
        " Configuration is as follows:\n" << str.str());
      success = false;
    }

    if (!(0 < test_exclusionary_radius_image_frac && test_exclusionary_radius_image_frac < 1.0)) 
    {
      std::stringstream str;
      config->print(str);
      LOG_ERROR(m_logger, "new_feat_exclusionary_radius_image_fraction (" 
        << test_exclusionary_radius_image_frac << ") must be between 0.0 and 1.0"
        " Configuration is as follows:\n" << str.str());
      success = false;
    }

    if (test_win_size < 3)
    {
      std::stringstream str;
      config->print(str);
      LOG_ERROR(m_logger, "win_size (" << test_win_size << ") must be three or more"
        " Configuration is as follows:\n" << str.str());
      success = false;
    }

    return success;
  }

  class featureDistributionImage {
  public:
    featureDistributionImage(): 
      rows(0), cols(0), 
      redetect_threshold(0.7),
      bad_bins_frac_to_redetect(0.125)
    {
      setGridSize(4, 4);
    }

    void setGridSize(int _rows, int _cols) {
      if (rows != _rows || cols != _cols) {
        rows = _rows;
        cols = _cols;
        distImage = cv::Mat(rows, cols, CV_8UC1);
        distImage.setTo(0);
      }
    }
    void getGridSize(int &_rows, int& _cols) const {
      _rows = rows;
      _cols = cols;
    }

    featureDistributionImage& operator=(const featureDistributionImage &other) {
      if (&other == this) {
        return *this;
      }
      other.distImage.copyTo(distImage);
    }

    bool shouldRedetect(const featureDistributionImage &lastDetectDist) {
      int badBins = 0;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          if (distImage.at<unsigned char>(r, c) < unsigned char(redetect_threshold*float(lastDetectDist.distImage.at<unsigned char>(r, c)))) {
            ++badBins;
          }
        }
      }
      if (badBins >= int(float(rows*cols)*bad_bins_frac_to_redetect)) {
        return true;
      }
      return false;
    }

    void setFromFeatureVector(const std::vector<cv::Point2f> &points, image_container_sptr image_data) {
      distImage.setTo(0);
      const int dist_bin_x_len = int(image_data->width() / cols);
      const int dist_bin_y_len = int(image_data->height() / rows);

      for (auto v = points.begin(); v != points.end(); ++v) {
        const cv::Point2f &tp = *v;
        int dist_bin_x = std::min<int>(std::max<int>(tp.x / dist_bin_x_len, 0), cols - 1);
        int dist_bin_y = std::min<int>(std::max<int>(tp.y / dist_bin_y_len, 0), rows - 1);

        unsigned char& numFeatInBin = distImage.at<unsigned char>(dist_bin_y, dist_bin_x);
        if (numFeatInBin < 255) {
          ++numFeatInBin;  //make sure we don't roll over the uchar.
        }
      }
    }
    
    cv::Mat distImage;    
    float redetect_threshold;
    float bad_bins_frac_to_redetect;    
  private:
      int rows, cols;      
  };

  kwiver::vital::logger_handle_t m_logger;
  featureDistributionImage distImage;
  featureDistributionImage lastDetect_distImage;
  int win_size;
  int half_win_size;
};


/// Default Constructor
track_features_klt
::track_features_klt()
: d_(new priv)
{
  attach_logger("ocv_track_features_klt");
  d_->m_logger = this->logger();
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

  d_->update_config(config);

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
  d_->set_config(config);
}


bool
track_features_klt
::check_configuration(vital::config_block_sptr config) const
{
  bool success(true);
  success = algo::detect_features::check_nested_algo_configuration("feature_detector", config) && success;
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
    throw vital::algorithm_configuration_exception(this->type_name(), this->impl_name(),
        "not all sub-algorithms have been initialized");
  }


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

  //setup stuff complete
  feature_track_set_sptr cur_tracks = prev_tracks;

  //points to be tracked in the next frame.  Empty at first.
  std::vector<cv::Point2f> next_points;

  //track features if there are any to track
  if (!d_->prev_points.empty())
  {
    //track
    std::vector<cv::Point2f> tracked_points;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(d_->prev_image, cv_img, d_->prev_points, tracked_points, status, err,
      cv::Size(d_->win_size, d_->win_size),3);

    std::vector<track_sptr> active_tracks = cur_tracks->active_tracks();  //gets the active tracks for the previous frame
    std::vector<feature_sptr> vf = cur_tracks->last_frame_features()->features();  //copy last frame's features   
    for (unsigned int i = 0; i< active_tracks.size(); ++i)
    {
      vector_2f tp(tracked_points[i].x, tracked_points[i].y);
      if (!status[i] || tp.x() <= d_->half_win_size || tp.y() <= d_->half_win_size 
        || tp.x() >= image_data->width()- d_->half_win_size 
        || tp.y() >= image_data->height()- d_->half_win_size)
      {
        //skip features that tracked to outside of the image (or the border) or didn't track properly
        continue;
      }
      auto f = std::make_shared<feature_f>(*vf[i]);  //info from feature detector (location, scale etc.)
      f->set_loc(tp);  //feature
      auto fts = std::make_shared<feature_track_state>(frame_number);  //feature, descriptor and frame number together
      fts->feature = f;
      track_sptr t = active_tracks[i];
      t->append(fts);  //append the feature's current location to it's track.  Track was picked up with active_tracks() call on previous_tracks.
      next_points.push_back(tracked_points[i]);
      //increment the feature distribution bins     
    }    
  }

  bool detectNewFeatures = next_points.size() <= size_t(d_->redetect_threshold*double(d_->lastDetectNumFeatures));  //did we track enough features from the previous frame?

  //set the feature distribution image
  d_->distImage.setFromFeatureVector(next_points, image_data);

  if (!detectNewFeatures){
    //now check the distribution of features in the image    
    if(d_->distImage.shouldRedetect(d_->lastDetect_distImage)){  //this will never be called on the first image so it will work.
      LOG_DEBUG(logger(), "detecting new feature because of distribution");
      detectNewFeatures = true;
    }
  }

  if( detectNewFeatures)
  {
    // detect new features
    // see if there are already existing tracks on this frame
    feature_set_sptr detected_feat;
    if(!cur_tracks)
    {
      cur_tracks = std::make_shared<kwiver::vital::feature_track_set>();
    }

    // get the last track id in the existing set of tracks and increment it
    track_id_t next_track_id = 0;
    if (!cur_tracks->all_track_ids().empty()) {
      next_track_id = (*cur_tracks->all_track_ids().crbegin()) + 1;
    }

    LOG_DEBUG( logger(), "Computing new features on frame "<<frame_number);
    // detect features on the current frame
    detected_feat = d_->detector->detect(image_data, mask);

    //merge new features into existing features (ignore new features near existing features)
    std::vector<feature_sptr> vf = detected_feat->features();

    // make a mask of current image feature positions. This maks keeps features from being 
    // kept that are detected near existing tracks.
    d_->setTrackedFeatureLocationMask(next_points, image_data);

    typedef std::vector<feature_sptr>::const_iterator feat_itr;    
    for(feat_itr fit = vf.begin(); fit != vf.end(); ++fit)
    {
      if (d_->trackedFeatureLocationMask.at<unsigned char>((*fit)->loc().y(), (*fit)->loc().x()) != 0) {
        continue;  //there is already a tracked feature near here
      }
      
      if ((*fit)->loc().x() < d_->win_size || 
          (*fit)->loc().y() < d_->win_size ||
          (*fit)->loc().x() > image_data->width() - d_->win_size ||
          (*fit)->loc().y() > image_data->height() - d_->win_size){
        continue;  //mask out features at the edge of the image.
      }

      auto fts = std::make_shared<feature_track_state>(frame_number);
      fts->feature = *fit;
      auto t = vital::track::create();  //make a new track
      t->append(fts);  //put the feature in this new track
      t->set_id(next_track_id++); //set the new track id
      cur_tracks->insert(t);
      next_points.push_back(cv::Point2f((*fit)->loc().x(), (*fit)->loc().y()));
    }
    d_->lastDetectNumFeatures = next_points.size();  //this includes any features tracked to this frame and the new points

    //store the last detected feature distribution
    d_->lastDetect_distImage.setFromFeatureVector(next_points, image_data);
  }

  //set up previous data structures for next call
  d_->prev_image = cv_img.clone();
  d_->prev_points = next_points;

  return cur_tracks;
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
