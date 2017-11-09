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
* \brief Header file for an abstract \link kwiver::vital::track_set track_set
*        \endlink and a concrete \link kwiver::vital::simple_track_set
*        simple_track_set \endlink
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

class VITAL_EXPORT keyframe_metadata {
public:
  virtual ~keyframe_metadata() {}
};

class VITAL_EXPORT keyframe_metadata_for_basic_selector :public keyframe_metadata
{
public:
  keyframe_metadata_for_basic_selector() = delete;

  keyframe_metadata_for_basic_selector(bool is_keyframe_)
    : is_keyframe(is_keyframe_) { }

  virtual ~keyframe_metadata_for_basic_selector() {}

  bool is_keyframe;
};

typedef std::shared_ptr<keyframe_metadata_for_basic_selector> keyframe_metadata_for_basic_selector_sptr;

typedef std::shared_ptr< keyframe_metadata> keyframe_metadata_sptr;

class keyframe_data;
/// Shared pointer for base keyframe_data type
typedef std::shared_ptr< keyframe_data > keyframe_data_sptr;
typedef std::shared_ptr< const keyframe_data > keyframe_data_const_sptr;

class simple_keyframe_data;
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

  virtual ~keyframe_data() {};

  virtual keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const = 0;

  virtual bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata) = 0;

  virtual bool remove_frame_metadata(frame_id_t frame) = 0;

  virtual keyframe_data_map_const_sptr get_keyframe_metadata_map() const = 0;

};

class VITAL_EXPORT simple_keyframe_data:public keyframe_data
{
public:
  simple_keyframe_data();

  virtual ~simple_keyframe_data();

  virtual keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const;

  virtual bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata);

  virtual bool remove_frame_metadata(frame_id_t frame);

  virtual keyframe_data_map_const_sptr get_keyframe_metadata_map() const;

protected:
  class priv;
  std::shared_ptr<priv> d_;
};

//this will go in a new kwiver arror shortly
class VITAL_EXPORT keyframe_data_graph
  :public keyframe_data
{
public:
  keyframe_data_graph();

  virtual ~keyframe_data_graph();

  virtual keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const;

  virtual bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata);

  virtual bool remove_frame_metadata(frame_id_t frame);

  virtual keyframe_data_map_const_sptr get_keyframe_metadata_map() const;

protected:
  class priv;
  std::shared_ptr<priv> d_;
  
};


} // end namespace vital
} // end namespace kwiver

#endif // VITAL_KEYFRAME_DATA_H_
