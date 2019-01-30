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
 * \brief Interface for match_descriptor_sets \link kwiver::vital::algo::algorithm_def
 *   algorithm definition \endlink.
 */

#ifndef VITAL_ALGO_MATCH_DESCRIPTOR_SETS_H_
#define VITAL_ALGO_MATCH_DESCRIPTOR_SETS_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/descriptor_set.h>
#include <vital/vital_types.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for matching sets of descriptors
/**
 * A common use for this algorithm is bag of visual words matching on sets of
 * descriptor extracted around features detected in images.
 */
class VITAL_ALGO_EXPORT match_descriptor_sets
  : public kwiver::vital::algorithm_def<match_descriptor_sets>
{
public:

  /// Desctuctor
  virtual ~match_descriptor_sets() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "match_descriptor_sets"; }

  /// Add a descriptor set to the inverted file system.
  /**
  * Add a descriptor set and frame number to the inverted file system.
  * Future matching results may include this frame in their results.
  *
  * \param[in] desc   set of descriptors associated with this frame
  * \param[in] frame  frame number indexing the descriptors
  * \returns None
  */
  virtual
  void
  append_to_index(const vital::descriptor_set_sptr desc,
                  vital::frame_id_t frame) = 0;

  /// Query the inverted file system for similar sets of descriptors.
  /**
  * Query the inverted file system and return the frames containing the most
  * similar sets descriptors.
  *
  * \param[in] desc  set of descriptors to match
  * \returns vector of possibly matching frames found by the query
  */
  virtual
  std::vector<vital::frame_id_t>
  query(const vital::descriptor_set_sptr desc) = 0;

  /// Query the inverted file system and append the descriptors.
  /**
  * This method is equivalent to calling query() followed by append_to_index();
  * however, depending on the implementation, it may be faster to call this
  * single function when both operations are required.
  *
  * \param[in] desc   set of descriptors to match and append
  * \param[in] frame  frame number indexing the descriptors
  * \returns vector of possibly matching frames found by the query
  */
  virtual
  std::vector<vital::frame_id_t>
  query_and_append(const vital::descriptor_set_sptr desc,
                   frame_id_t frame);

protected:

  /// Default constructor
  match_descriptor_sets();

};


/// Shared pointer type for generic image_io definition type.
typedef std::shared_ptr<match_descriptor_sets> match_descriptor_sets_sptr;


} } } // end namespace

#endif // VITAL_ALGO_MATCH_DESCRIPTOR_SETS_H_
