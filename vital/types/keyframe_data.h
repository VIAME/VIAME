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
* \brief  Classes to store metadata about frames.
*/

#ifndef VITAL_KEYFRAME_DATA_H_
#define VITAL_KEYFRAME_DATA_H_

#include "track.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vector>
#include <set>
#include <map>
#include <memory>

namespace kwiver {
namespace vital {

class keyframe_metadata;
typedef std::shared_ptr< keyframe_metadata> keyframe_metadata_sptr;

///  Virtual base class for keyframe metadata.
/**
* Interface for keyframe metadata.  At a minimum it is cloneable.
*/

class VITAL_EXPORT keyframe_metadata {
public:
  /// Desructor
  virtual ~keyframe_metadata() {}

  // Clone the metadata
  virtual keyframe_metadata_sptr clone() const = 0;
};

class keyframe_metadata_for_basic_selector;
typedef std::shared_ptr<keyframe_metadata_for_basic_selector> keyframe_metadata_for_basic_selector_sptr;

/// Stores if a frame is a keyframe or not.
/**
* Stores the metadata of whether or not a frame is a keyframe.
*/

class VITAL_EXPORT keyframe_metadata_for_basic_selector :public keyframe_metadata
{
public:

  /// Constructor
  keyframe_metadata_for_basic_selector() = delete;

  /// Returns true iff the frame is a keyframe
  keyframe_metadata_for_basic_selector(bool is_keyframe_)
    : is_keyframe(is_keyframe_) { }

  /// Destructor
  virtual ~keyframe_metadata_for_basic_selector() {}

  /// Clone this metadata and return a pointer to it
  virtual keyframe_metadata_sptr clone() const
  {
    return std::make_shared< keyframe_metadata_for_basic_selector >(this->is_keyframe);
  }

  /// True iff the frame is a keyframe
  bool is_keyframe;
};

class keyframe_data;
/// Shared pointer for base keyframe_data type
typedef std::shared_ptr< keyframe_data > keyframe_data_sptr;
typedef std::shared_ptr< const keyframe_data > keyframe_data_const_sptr;

class simple_keyframe_data;
/// Shared pointer for base simple_keyframe_data type
typedef std::shared_ptr< simple_keyframe_data> simple_keyframe_data_sptr;
typedef std::shared_ptr< const simple_keyframe_data> simple_keyframe_data_const_sptr;

typedef std::map<frame_id_t, keyframe_metadata_sptr> keyframe_data_map;
typedef std::shared_ptr<keyframe_data_map> keyframe_data_map_sptr;
typedef std::shared_ptr<const keyframe_data_map> keyframe_data_map_const_sptr;

/// A collection of keyframes
/**
* This class is a very basic keyframe data structure.  We can do better
* with a graph etc.
*/
class VITAL_EXPORT keyframe_data
{
public:

  /// Destructor
  virtual ~keyframe_data() {};

  /// Get a pointer to the metadata for a particular frame
  virtual keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const = 0;

  /// Set the metadata for a frame
  virtual bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata) = 0;

  /// Remove the metadata for a pariticular frame from the data structure
  virtual bool remove_frame_metadata(frame_id_t frame) = 0;

  /// Get a pointer to the keyframe_data_map that stores all of the keyframe metadata
  virtual keyframe_data_map_const_sptr get_keyframe_metadata_map() const = 0;

  /// Clone this keyframe data
  virtual keyframe_data_sptr clone() const = 0;

};

class VITAL_EXPORT simple_keyframe_data:public keyframe_data
{
public:

  /// Constructor
  simple_keyframe_data();

  /// Destructor
  virtual ~simple_keyframe_data();

  /// Get the metadata for a frame
  virtual keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const;

  /// Set the metadata for a frame
  virtual bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata);

  /// Remove the metadata for a frame from the data structure
  virtual bool remove_frame_metadata(frame_id_t frame);

  /// Get a pointer to the keyframe_data_map that stores all of the keyframe metadata
  virtual keyframe_data_map_const_sptr get_keyframe_metadata_map() const;

  /// Clone this keyframe data
  virtual keyframe_data_sptr clone() const;

protected:
  class priv;
  std::shared_ptr<priv> d_;
};

} // end namespace vital
} // end namespace kwiver

#endif // VITAL_KEYFRAME_DATA_H_
