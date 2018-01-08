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

#ifndef VITAL_ALGO_BAG_OF_WORDS_MATCHING_H_
#define VITAL_ALGO_BAG_OF_WORDS_MATCHING_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/descriptor_set.h>
#include <vital/vital_types.h>


namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for bag of words image matching
/**
 * This class represents an abstract interface for bag of words image matching
 */
class VITAL_ALGO_EXPORT bag_of_words_matching
  : public kwiver::vital::algorithm_def<bag_of_words_matching>
{
public:

  /// Desctuctor
  virtual ~bag_of_words_matching() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "bag_of_words_matching"; }

  /// Add an image to the inverted file system.
  /**
  * Add the image to the inverted file system.  Future matching results may
  * include this image in their results.
  * \param[in] desc set of descriptors for the image
  * \param[in] frame_number frame of the associated image
  * \returns None
  */
  virtual
  void
  append_to_index(const vital::descriptor_set_sptr desc,
                  vital::frame_id_t frame_number) = 0;

  /// Query the inverted file system for similar images.
  /**
  * Query the inverted file system and return the most similar images.
  * \param[in] desc set of descriptors for the image
  * \returns vector of possibly matching frames found by the query
  */
  virtual
  std::vector<vital::frame_id_t>
  query(const vital::descriptor_set_sptr desc) = 0;

  /// Query the inverted file system for similar images and append the querying image.
  /**
  * Query the inverted file system and return the most similar images.  This method
  * may be faster than first querying and then appending if both operations are required.
  * \param[in] desc set of descriptors for the image
  * \param[in] frame id of the query image
  * \returns vector of possibly matching frames found by the query
  */

  virtual
  std::vector<vital::frame_id_t>
  query_and_append(const vital::descriptor_set_sptr desc,
                   frame_id_t frame);

protected:

  /// Default constructor
  bag_of_words_matching();

};


/// Shared pointer type for generic image_io definition type.
typedef std::shared_ptr<bag_of_words_matching> bag_of_words_matching_sptr;


} } } // end namespace

#endif // VITAL_ALGO_BAG_OF_WORDS_MATCHING_H_
