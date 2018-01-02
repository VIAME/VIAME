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
 * \brief Interface for bag_of_words_matching \link kwiver::vital::algo::algorithm_def 
 *   algorithm definition \endlink.
 */

#ifndef VITAL_DBOW2_BAG_OF_WORDS_MATCHING_H_
#define VITAL_DBOW2_BAG_OF_WORDS_MATCHING_H_

#include <vital/vital_config.h>

#include <arrows/DBow2/kwiver_algo_DBoW2_export.h>

#include <vital/algo/bag_of_words_matching.h>


namespace kwiver {
namespace arrows {
namespace DBoW2_kw {

/// An abstract base class for reading and writing images
/**
 * This class represents an abstract interface for bag of words image matching
 */
class KWIVER_ALGO_DBOW2_EXPORT bag_of_words_matching
  : public vital::algorithm_impl<bag_of_words_matching, vital::algo::bag_of_words_matching>
{
public:

  bag_of_words_matching();

  virtual ~bag_of_words_matching();

  virtual void append_to_index( const vital::descriptor_set_sptr desc, 
                                vital::frame_id_t frame_number);

  virtual void query( const vital::descriptor_set_sptr, 
                      vital::frame_id_t frame_number, 
                      std::vector<vital::frame_id_t> &putative_matching_frames,
                      bool append_to_index_on_query);
  
  virtual vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( vital::config_block_sptr config);
  
  virtual bool check_configuration( vital::config_block_sptr config) const;

protected:
  /// the feature m_detector algorithm
  class priv;
  std::shared_ptr<priv> d_;

};


/// Shared pointer type for generic image_io definition type.
typedef std::shared_ptr<bag_of_words_matching> bag_of_words_matching_sptr;

} } } // end namespace

#endif // VITAL_DBOW2_BAG_OF_WORDS_MATCHING_H_
