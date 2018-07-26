/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief Implementation of bag of words matching
 */

#include "arrows/dbow2/match_descriptor_sets.h"
#include "DBoW2.h"
#include <opencv2/features2d.hpp>


#include <vital/logger/logger.h>
#include <vital/algo/algorithm.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/image_io.h>
#include <vital/algo/match_features.h>
#include <kwiversys/SystemTools.hxx>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace dbow2 {

class match_descriptor_sets::priv
{
public:
  priv();

  void train_vocabulary(
    std::string training_image_list,
    std::string vocabulary_output_file);

  void train(
    std::vector<std::vector<cv::Mat > > const &features,
    std::string voc_file_path);

  void setup_voc();

  void load_vocabulary(std::string voc_file_path);

  void load_features(std::string training_image_list,
    std::vector<std::vector<cv::Mat > > &features);

  void descriptor_set_to_vec(
    descriptor_set_sptr im_descriptors,
    std::vector<cv::Mat> &features) const;

  cv::Mat descriptor_to_mat(descriptor_sptr) const;

  std::vector<frame_id_t>
  query(kwiver::vital::descriptor_set_sptr desc,
        frame_id_t frame_number,
        bool append_to_index_on_query);

  void append_to_index(const vital::descriptor_set_sptr desc,
                       vital::frame_id_t frame);

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

  // The path to the training image list
  std::string training_image_list_path;

  // The path to the vocabulary
  std::string vocabulary_path;

  std::map<DBoW2::EntryId, kwiver::vital::frame_id_t> m_entry_to_frame;

  int m_max_num_candidate_matches_from_vocabulary_tree;
};

//-----------------------------------------------------------------------------

match_descriptor_sets::priv
::priv()
  :training_image_list_path("")
  ,vocabulary_path("kwiver_voc.yml.gz")
  ,m_max_num_candidate_matches_from_vocabulary_tree(10)
{

}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::setup_voc()
{
  if (!m_voc)
  {
    //first time we will make the voc.  Then just load it.
    try {
      load_vocabulary(vocabulary_path);
    }
    catch (const path_not_a_file &e)
    {
      LOG_DEBUG(m_logger, e.what());
      m_voc.reset();
    }
    catch (const path_not_exists &e)
    {
      LOG_DEBUG(m_logger, e.what());
      m_voc.reset();
    }

    if (!m_voc)
    {
      train_vocabulary(training_image_list_path, vocabulary_path);
    }

    m_db = std::make_shared<OrbDatabase>(*m_voc, true, 3);
  }
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::append_to_index(const vital::descriptor_set_sptr desc,
  vital::frame_id_t frame_number)
{
  setup_voc();

  if (!desc)
  {
    return;
  }

  std::vector<cv::Mat> desc_mats;
  descriptor_set_to_vec(desc, desc_mats);  // note that desc_mats can be shorter
                                           // than desc because of null
                                           // descriptors (KLT features)

  if (desc_mats.size() == 0)
  {  //only features without descriptors in this frame
    return;
  }

  //run them through the vocabulary to get the BOW vector
  DBoW2::BowVector bow_vec;
  DBoW2::FeatureVector feat_vec;
  m_voc->transform(desc_mats, bow_vec, feat_vec, 3);

  const DBoW2::EntryId ent = m_db->add(bow_vec, feat_vec);
  std::pair<const DBoW2::EntryId, kwiver::vital::frame_id_t>
    new_ent(ent, frame_number);

  m_entry_to_frame.insert(new_ent);
}

//-----------------------------------------------------------------------------

std::vector<frame_id_t>
match_descriptor_sets::priv
::query( kwiver::vital::descriptor_set_sptr desc,
         frame_id_t frame_number,
         bool append_to_index_on_query)
{
  setup_voc();

  std::vector<frame_id_t> putative_matches;

  if (!desc)
  {
    return putative_matches;
  }

  std::vector<cv::Mat> desc_mats;
  descriptor_set_to_vec(desc, desc_mats);  // note that desc_mats can be shorter
                                           // than desc because of null
                                           // descriptors (KLT features)

  if (desc_mats.size() == 0)
  {  //only features without descriptors in this frame
    return putative_matches;
  }

  //run them through the vocabulary to get the BOW vector
  DBoW2::BowVector bow_vec;
  DBoW2::FeatureVector feat_vec;
  m_voc->transform(desc_mats, bow_vec, feat_vec, 3);

  int max_res = m_max_num_candidate_matches_from_vocabulary_tree;
  DBoW2::QueryResults ret;

  //add them to the database
  if (append_to_index_on_query)
  {
    const DBoW2::EntryId ent = m_db->add(bow_vec, feat_vec);
    std::pair<const DBoW2::EntryId, kwiver::vital::frame_id_t>
      new_ent(ent, frame_number);

    m_entry_to_frame.insert(new_ent);

    //querry the database for matches

    m_db->query(bow_vec, ret, max_res, ent);  // ent at the end prevents the
                                              // querry from returning the
                                              // current image.
  }
  else
  {
    m_db->query(bow_vec, ret, max_res);
  }

  putative_matches.reserve(ret.size());

  for (auto r : ret)
  {
    auto put_match = m_entry_to_frame.find(r.Id);
    if (put_match == m_entry_to_frame.end())
    {
      continue;
    }
    putative_matches.push_back(put_match->second);
  }

  return putative_matches;
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::train_vocabulary(
  std::string training_image_list,
  std::string vocabulary_output_file)
{
  std::vector<std::vector<cv::Mat > > features;
  load_features(training_image_list, features);

  train(features, vocabulary_output_file);
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::train(
  std::vector<std::vector<cv::Mat > > const &features,
  std::string voc_file_path)
{
  const int k = 10;  //branching factor
  const int L = 6;   //number of levels
  const DBoW2::WeightingType weight = DBoW2::TF_IDF;
  const DBoW2::ScoringType score = DBoW2::L1_NORM;

  m_voc = std::make_shared<OrbVocabulary>(k, L, weight, score);
  m_voc->create(features);

  // save the vocabulary to disk
  LOG_INFO(m_logger, "Saving vocabulary ...");
  m_voc->save(voc_file_path);
  LOG_INFO(m_logger, "Done saving vocabulary");
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::load_vocabulary(std::string voc_file_path)
{
  if (!kwiversys::SystemTools::FileExists(voc_file_path))
  {
    throw path_not_exists(voc_file_path);
  }
  else if (kwiversys::SystemTools::FileIsDirectory(voc_file_path))
  {
    throw path_not_a_file(voc_file_path);
  }


  m_voc = std::make_shared<OrbVocabulary>(voc_file_path);
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
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
    throw vital::invalid_file(training_image_list,
      "unable to open training image file");
  }

  int ln_num = 0;
  while (std::getline(im_list, line))
  {
    if (ln_num++ != 10)
    {
      continue;
    }
    ln_num = 0;
    image_container_sptr im = m_image_io->load(line);
    LOG_INFO(m_logger, "Extracting features for image " + line);

    feature_set_sptr im_features = m_detector->detect(im);
    descriptor_set_sptr im_descriptors = m_extractor->extract(im, im_features);

    features.push_back(std::vector<cv::Mat >());
    descriptor_set_to_vec(im_descriptors, features.back());
  }

  if (im_list.bad())
  {
    LOG_ERROR(m_logger, "error while reading " + training_image_list);
    throw vital::invalid_file(training_image_list, "training image list bad");
  }
}

//-----------------------------------------------------------------------------

void
match_descriptor_sets::priv
::descriptor_set_to_vec(
  descriptor_set_sptr im_descriptors,
  std::vector<cv::Mat> &features) const
{
  std::vector< descriptor_sptr > desc = im_descriptors->descriptors();
  features.resize(desc.size());
  unsigned int dn = 0;
  for (auto d : desc)
  {
    if (!d)
    {
      //skip null descriptors
      continue;
    }
    features[dn++] = descriptor_to_mat(d);
  }

  features.resize(dn);  //resize to only return features for non-null descriptors
}

//-----------------------------------------------------------------------------

cv::Mat
match_descriptor_sets::priv
::descriptor_to_mat(descriptor_sptr desc) const
{
  std::vector<kwiver::vital::byte> desc_bytes = desc->as_bytes();
  cv::Mat desc_mat = cv::Mat(1, static_cast<int>(desc_bytes.size()), CV_8UC1);
  unsigned int bn = 0;
  for (auto b : desc_bytes)
  {
    desc_mat.at<unsigned char>(0, bn++) = b;
  }
  return desc_mat;
}

//-----------------------------------------------------------------------------

match_descriptor_sets
::match_descriptor_sets()
 : d_(new priv)
{
  attach_logger("arrows.dbow2.match_descriptor_sets");
  d_->m_logger = this->logger();
}

match_descriptor_sets
::~match_descriptor_sets()
{

}

void
match_descriptor_sets
::append_to_index(const descriptor_set_sptr desc, frame_id_t frame_number)
{
  d_->append_to_index(desc, frame_number);
}

std::vector<frame_id_t>
match_descriptor_sets
::query( const descriptor_set_sptr desc )
{
  return d_->query(desc,-1,false);
}

std::vector<frame_id_t>
match_descriptor_sets
::query_and_append( const vital::descriptor_set_sptr desc,
                    frame_id_t frame)
{
  return d_->query(desc, frame, true);
}

// ------------------------------------------------------------------

vital::config_block_sptr
match_descriptor_sets::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm
  algo::detect_features::
    get_nested_algo_configuration("feature_detector", config, d_->m_detector);

  // - Descriptor Extractor algorithm
  algo::extract_descriptors::
    get_nested_algo_configuration("descriptor_extractor", config, d_->m_extractor);

  algo::image_io::
    get_nested_algo_configuration("image_io", config, d_->m_image_io);

  config->set_value("max_num_candidate_matches_from_vocabulary_tree",
    d_->m_max_num_candidate_matches_from_vocabulary_tree,
    "the maximum number of candidate matches to return from the vocabulary tree");

  config->set_value("training_image_list_path",
    d_->training_image_list_path,
    "path to the list of vocabulary training images");

  config->set_value("vocabulary_path",
    d_->vocabulary_path,
    "path to the vocabulary file");

  return config;
}

// ------------------------------------------------------------------

void
match_descriptor_sets::
set_configuration(vital::config_block_sptr config_in)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config(config_in);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::detect_features_sptr df;
  algo::detect_features::set_nested_algo_configuration(
    "feature_detector", config, df);
  d_->m_detector = df;

  algo::extract_descriptors_sptr ed;
  algo::extract_descriptors::set_nested_algo_configuration(
    "descriptor_extractor", config, ed);
  d_->m_extractor = ed;

  algo::image_io_sptr io;
  algo::image_io::set_nested_algo_configuration("image_io", config, io);
  d_->m_image_io = io;

  d_->m_max_num_candidate_matches_from_vocabulary_tree =
    config->get_value<int>("max_num_candidate_matches_from_vocabulary_tree",
      d_->m_max_num_candidate_matches_from_vocabulary_tree);

  d_->training_image_list_path =
    config->get_value<std::string>("training_image_list_path", d_->training_image_list_path);

  d_->vocabulary_path =
    config->get_value<std::string>("vocabulary_path", d_->vocabulary_path);
}

// ------------------------------------------------------------------

bool
match_descriptor_sets::
check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid =
    algo::detect_features::check_nested_algo_configuration(
      "feature_detector", config) && config_valid;

  config_valid =
    algo::extract_descriptors::check_nested_algo_configuration(
      "descriptor_extractor", config) && config_valid;

  config_valid =
    algo::image_io::check_nested_algo_configuration("image_io", config) &&
    config_valid;

  int max_cand_matches =
    config->get_value<int>("max_num_candidate_matches_from_vocabulary_tree");

  if (max_cand_matches <= 0)
  {
    LOG_ERROR(d_->m_logger,
      "max_num_candidate_matches_from_vocabulary_tree must be a positive "
      "(nonzero) integer");
    config_valid = false;
  }

  return config_valid;
}

} } } // end namespace
