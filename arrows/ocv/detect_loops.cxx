/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief OCV detect_loops algorithm implementation
 */

#include <map>

#include "detect_loops.h"


using namespace kwiver::vital;

#include <DBoW2\DBoW2.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <vital/logger/logger.h>
#include <vital/algo/algorithm.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/image_io.h>
#include <vital/algo/match_features.h>

using namespace DBoW2;
using namespace DUtils;

namespace kwiver {
namespace arrows {
namespace ocv {

class detect_loops::priv 
{
public:
  priv();
  
  void query_vocab(frame_id_t frame_number, 
                   std::vector<frame_id_t> &putative_matches);

  void train_vocabulary(
    std::string training_image_list,
    std::string vocabulary_output_file);

  void train(
    std::vector<std::vector<cv::Mat > > const &features,
    std::string voc_file_path);

  void load_vocabulary(std::string voc_file_path);

  void load_features(std::string training_image_list,
    std::vector<std::vector<cv::Mat > > &features);  

  kwiver::vital::feature_track_set_sptr
    detect(kwiver::vital::feature_track_set_sptr feat_tracks,
      kwiver::vital::frame_id_t frame_number);

  kwiver::vital::feature_track_set_sptr
    verify_and_add_image_matches(
      kwiver::vital::feature_track_set_sptr feat_tracks,
      kwiver::vital::frame_id_t frame_number,
      std::vector<frame_id_t> const &putative_matches);

  cv::Mat descriptor_to_mat(descriptor_sptr) const;

  void descriptor_set_to_vec(
    descriptor_set_sptr im_descriptors,
    std::vector<cv::Mat> &features) const;

  kwiver::vital::logger_handle_t m_logger;
  
  // The vocabulary tree
  std::shared_ptr<OrbVocabulary> m_voc;
  
  // The inverted file database
  std::shared_ptr<OrbDatabase> m_db;
  
  /// The feature m_detector algorithm to use
  vital::algo::detect_features_sptr m_detector;
  
  /// The descriptor extractor algorithm to use
  vital::algo::extract_descriptors_sptr m_extractor;
  
  // The image io to use
  vital::algo::image_io_sptr m_image_io;
  
  /// The feature matching algorithm to use
  vital::algo::match_features_sptr m_matcher;

  /// The loop closure algorithm to use
  //vital::algo::close_loops_sptr closer;

  std::map<DBoW2::EntryId, kwiver::vital::frame_id_t> m_entry_to_frame;  

};

//-----------------------------------------------------------------------------

detect_loops::priv
::priv()
{
  m_db = std::make_shared<OrbDatabase>(false,0);
}

//-----------------------------------------------------------------------------

void
detect_loops::priv
::query_vocab(frame_id_t frame_number,
  std::vector<frame_id_t> &putative_matches)
{

}

//-----------------------------------------------------------------------------

void 
detect_loops::priv
::train_vocabulary(
  std::string training_image_list,
  std::string vocabulary_output_file)
{
  std::vector<std::vector<cv::Mat > > features;
  load_features(training_image_list,features);

  train(features,vocabulary_output_file);
}

//-----------------------------------------------------------------------------

void 
detect_loops::priv
::train(
  std::vector<std::vector<cv::Mat > > const &features,
  std::string voc_file_path)
{  
  const int k = 10;  //branching factor
  const int L = 6;   //number of levels
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  m_voc = std::make_shared<OrbVocabulary>(k, L, weight, score);  
  m_voc->create(features);  

  // save the vocabulary to disk  
  LOG_INFO(m_logger, "Saving vocabulary ...");  
  m_voc->save(voc_file_path);
  LOG_INFO(m_logger, "Done saving vocabulary");
}

//-----------------------------------------------------------------------------

void 
detect_loops::priv
::load_vocabulary(std::string voc_file_path)
{
  m_voc = std::make_shared<OrbVocabulary>(voc_file_path);
}

//-----------------------------------------------------------------------------

void 
detect_loops::priv
::load_features(
  std::string training_image_list, 
  std::vector<std::vector<cv::Mat > > &features)
{
  features.clear();
  features.reserve(100);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  std::ifstream im_list;
  im_list.open(training_image_list);

  std::cout << "Extracting ORB features..." << std::endl;

  std::string line;
  if (!im_list.is_open())
  {
    LOG_ERROR(m_logger, "error while opening file " + training_image_list);
  }

  while (std::getline(im_list, line)) 
  {   
   
    image_container_sptr im = m_image_io->load(line);

    feature_set_sptr im_features = m_detector->detect(im);
    descriptor_set_sptr im_descriptors = m_extractor->extract(im, im_features);

    features.push_back(std::vector<cv::Mat >());
    descriptor_set_to_vec(im_descriptors, features.back());   
  }

  if (im_list.bad())
  {
    LOG_ERROR(m_logger, "error while reading " + training_image_list);
  }
}

//-----------------------------------------------------------------------------

void 
detect_loops::priv
::descriptor_set_to_vec(
  descriptor_set_sptr im_descriptors, 
  std::vector<cv::Mat> &features) const
{
  std::vector< descriptor_sptr > desc = im_descriptors->descriptors();
  features.resize(desc.size());
  unsigned int dn = 0;
  for (auto d : desc)
  {
    features[dn++] = descriptor_to_mat(d);
  }
}


//-----------------------------------------------------------------------------

cv::Mat 
detect_loops::priv
::descriptor_to_mat(descriptor_sptr desc) const
{
  std::vector<kwiver::vital::byte> desc_bytes = desc->as_bytes();
  cv::Mat desc_mat = cv::Mat(1, desc_bytes.size(), CV_8UC1);
  unsigned int bn = 0;
  for (auto b : desc_bytes)
  {
    desc_mat.at<unsigned char>(0, bn++) = b;
  }
  return desc_mat;
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
detect_loops::priv
::verify_and_add_image_matches(
  kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number, 
  std::vector<frame_id_t> const &putative_matches)
{
  feature_set_sptr feat1, feat2;
  descriptor_set_sptr desc1, desc2;

  feat1 = feat_tracks->frame_features(frame_number);
  desc1 = feat_tracks->frame_descriptors(frame_number);

  feature_track_set_sptr loop_connected_tracks = std::dynamic_pointer_cast<feature_track_set>(feat_tracks->clone());
  std::vector<kwiver::vital::track_sptr> tracks1 = loop_connected_tracks->active_tracks(frame_number);

  for (auto fn2 : putative_matches)
  {
    if (fn2 == frame_number)
    {
      continue; // no sense matching an image to itself
    }
    feat2 = feat_tracks->frame_features(fn2);
    desc2 = feat_tracks->frame_descriptors(fn2);

    match_set_sptr mset = m_matcher->match(feat1, desc1, feat2, desc2);
    if (!mset)
    {
      LOG_WARN(m_logger, "Feature matching between frames " << frame_number <<
        " and " << fn2 << " failed");
      continue;
    }

    std::vector<track_sptr> tracks2 = feat_tracks->active_tracks(fn2);
    
    std::vector<match> vm = mset->matches();

      int num_linked = 0;
      for (match m : vm)
      {
        track_sptr t1 = tracks1[m.first];
        track_sptr t2 = tracks2[m.second];
        if (loop_connected_tracks->merge_tracks(t1, t2))
        {
          ++num_linked;
        }
      }
      LOG_DEBUG(m_logger, "Stitched " << num_linked <<
        " tracks between frames " << frame_number <<
        " and " << fn2);
  }

  return loop_connected_tracks;
}
  


//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
detect_loops::priv
::detect(kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number)
{
  //get the feature descriptors in the current image
 
  descriptor_set_sptr desc = feat_tracks->frame_descriptors(frame_number);
  if (!desc)
  {
    return feat_tracks;
  }
  std::vector<cv::Mat> desc_mats;
  descriptor_set_to_vec(desc, desc_mats);

  //run them through the vocabulary to get the BOW vector
  DBoW2::BowVector bow_vec;
  DBoW2::FeatureVector feat_vec;
  m_voc->transform(desc_mats, bow_vec, feat_vec, 3);

  //add them to the database
  const DBoW2::EntryId ent = m_db->add(bow_vec, feat_vec);
  std::pair<const DBoW2::EntryId, kwiver::vital::frame_id_t> new_ent(ent, frame_number);
  m_entry_to_frame.insert(new_ent);

  //querry the database for matches
  int max_res = 50;
  DBoW2::QueryResults ret;
  m_db->query(bow_vec, ret, max_res, ent);  //ent at the end prevents the querry from returning the current image.

  std::vector<frame_id_t> putative_matching_images;
  putative_matching_images.reserve(ret.size());

  for (auto r : ret)
  {
    auto put_match = m_entry_to_frame.find(r.Id);
    if (put_match == m_entry_to_frame.end())
    {
      continue;
    }
    putative_matching_images.push_back(put_match->second);
  }

  return verify_and_add_image_matches(feat_tracks, frame_number, putative_matching_images);
}

// ----------------------------------------------------------------------------

detect_loops
::detect_loops()
{
  d_ = std::make_shared<priv>();
  attach_logger("detect_loops");
  d_->m_logger = this->logger();

  std::string voc_file = "kwiver_voc.yml.gz";
  //first time we will make the voc.  Then just load it.
  bool make_voc = true;
  if (make_voc)
  {
    d_->train_vocabulary("training_image_list.txt", voc_file);
  }
  else
  {
    d_->load_vocabulary(voc_file);
  } 
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
detect_loops
::detect(kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number)
{

  return d_->detect(feat_tracks, frame_number);
}

//-----------------------------------------------------------------------------

void 
detect_loops
::train_vocabulary(
  std::string training_image_path,
  std::string vocabulary_output_file)
{
  d_->train_vocabulary(training_image_path, vocabulary_output_file);
}

//-----------------------------------------------------------------------------

void 
detect_loops
::load_vocabulary(
  std::string vocabulary_file)
{
  d_->load_vocabulary(vocabulary_file);
}

//-----------------------------------------------------------------------------

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_loops
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm
  algo::detect_features::
    get_nested_algo_configuration("feature_detector", config, d_->m_detector);

  // - Descriptor Extractor algorithm
  algo::extract_descriptors::
    get_nested_algo_configuration("descriptor_extractor", config, d_->m_extractor);

  algo::image_io::
    get_nested_algo_configuration("image_io", config, d_->m_image_io);

  algo::match_features::
    get_nested_algo_configuration("match_features", config, d_->m_matcher);

  return config;
}

//-----------------------------------------------------------------------------

/// Set this algo's properties via a config block
void
detect_loops
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
  d_->m_detector = df;

  algo::extract_descriptors_sptr ed;
  algo::extract_descriptors::set_nested_algo_configuration("descriptor_extractor", config, ed);
  d_->m_extractor = ed;

  algo::image_io_sptr io;
  algo::image_io::set_nested_algo_configuration("image_io", config, io);
  d_->m_image_io = io;

  algo::match_features_sptr mf;
  algo::match_features::set_nested_algo_configuration("match_features", config, mf);
  d_->m_matcher = mf;


}

//-----------------------------------------------------------------------------

bool
detect_loops
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid = algo::detect_features::check_nested_algo_configuration("feature_detector", config)
    && config_valid;

  config_valid = algo::extract_descriptors::check_nested_algo_configuration("descriptor_extractor", config)
    && config_valid;

  config_valid = algo::image_io::check_nested_algo_configuration("image_io", config) && config_valid;

  config_valid = algo::match_features::check_nested_algo_configuration("match_features", config) && config_valid;
    
  return config_valid;
}

//-----------------------------------------------------------------------------

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
