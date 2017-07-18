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
 * \brief Header file for \link kwiver::vital::feature_track_set feature_track_set
 *        \endlink and a concrete \link kwiver::vital::simple_feature_track_set
 *        simple_feature_track_set \endlink
 */

#ifndef VITAL_FEATURE_TRACK_SET_H_
#define VITAL_FEATURE_TRACK_SET_H_

#include "descriptor_set.h"
#include "feature_set.h"
#include "track_set.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vector>
#include <memory>

namespace kwiver {
namespace vital {


/// A derived track_state_data for feature tracks
class VITAL_EXPORT feature_track_state_data : public track_state_data
{
public:
  /// Constructor
  feature_track_state_data( feature_sptr f,
                            descriptor_sptr d )
  : feature( f ),
    descriptor( d ) {}

  feature_track_state_data() {}

  feature_sptr feature;
  descriptor_sptr descriptor;
};


class feature_track_set;
/// Shared pointer for feature_track_set type
typedef std::shared_ptr< feature_track_set > feature_track_set_sptr;

/// A collection of 2D feature point tracks
class VITAL_EXPORT feature_track_set : public track_set
{
public:
  /// Destructor
  virtual ~feature_track_set() VITAL_DEFAULT_DTOR

  /// Return the set of features in tracks on the last frame
  virtual feature_set_sptr last_frame_features() const;

  /// Return the set of descriptors in tracks on the last frame
  virtual descriptor_set_sptr last_frame_descriptors() const;

  /// Return the set of features in all tracks for the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a feature_set_sptr for all features on the give frame.
   */
  virtual feature_set_sptr frame_features( frame_id_t offset = -1 ) const;

  /// Return the set of descriptors in all tracks for the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a descriptor_set_sptr for all features on the give frame.
   */
  virtual descriptor_set_sptr frame_descriptors( frame_id_t offset = -1 ) const;

};


/// A concrete feature track set that simply wraps a vector of tracks.
class simple_feature_track_set :
  public feature_track_set
{
public:
  /// Default Constructor
  simple_feature_track_set() VITAL_DEFAULT_CTOR

  /// Constructor from a vector of tracks
  explicit simple_feature_track_set( const std::vector< track_sptr >& tracks )
    : data_( tracks ) { }

  /// Return the number of tracks in the set
  virtual size_t size() const { return data_.size(); }

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const { return data_; }


protected:
  /// The vector of tracks
  std::vector< track_sptr > data_;
};


} } // end namespace vital

#endif // VITAL_FEATURE_TRACK_SET_H_
