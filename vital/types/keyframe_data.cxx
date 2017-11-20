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
* \brief Implementation of keyframe_data
*/

#include "keyframe_data.h"
#include <map>

namespace kwiver {
namespace vital {
  
class simple_keyframe_data::priv
{
public:
  priv()
  {
    kf_map_ptr = std::make_shared<keyframe_data_map>();
  }

  keyframe_metadata_sptr get_frame_metadata(frame_id_t frame) const
  {
    auto kfm_it = kf_map_ptr->find(frame);
    if (kfm_it == kf_map_ptr->end())
    {
      return keyframe_metadata_sptr();
    }
    else
    {
      return kfm_it->second;
    }
  }

  bool set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata)
  {
    (*kf_map_ptr)[frame] = metadata;
    return true;
  }

  virtual bool remove_frame_metadata(frame_id_t frame)
  {
    auto kfm_it = kf_map_ptr->find(frame);
    if ( kfm_it != kf_map_ptr->end())
    {
      kf_map_ptr->erase(kfm_it);
      return true;
    }
    return false;
  }

  keyframe_data_map_const_sptr get_keyframe_metadata_map() const
  {
    return kf_map_ptr;
  }

  std::shared_ptr<simple_keyframe_data::priv> clone() const
  {
    std::shared_ptr<simple_keyframe_data::priv> new_skd_priv =
      std::make_shared<simple_keyframe_data::priv>();
    
    for (auto kf : *kf_map_ptr)
    {
      (*(new_skd_priv->kf_map_ptr))[kf.first] = kf.second->clone();
    }
    return new_skd_priv;
  }

  keyframe_data_map_sptr kf_map_ptr;
};  //end keframe_data::priv 

    //-------------------------------------------------------------------------

simple_keyframe_data
::simple_keyframe_data()
{
  d_ = std::make_shared<simple_keyframe_data::priv>();
}

simple_keyframe_data
::~simple_keyframe_data()
{

}

keyframe_data_sptr 
simple_keyframe_data
::clone() const
{
  simple_keyframe_data_sptr new_skd = std::make_shared<simple_keyframe_data>();
  new_skd->d_ = this->d_->clone();
  return new_skd;
}

keyframe_metadata_sptr
simple_keyframe_data
::get_frame_metadata(frame_id_t frame) const
{
  return d_->get_frame_metadata(frame);
}

bool 
simple_keyframe_data
::set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata)
{
  return d_->set_frame_metadata(frame, metadata);
}

bool 
simple_keyframe_data
::remove_frame_metadata(frame_id_t frame)
{
  return d_->remove_frame_metadata(frame);
}

keyframe_data_map_const_sptr 
simple_keyframe_data
::get_keyframe_metadata_map() const
{
  return d_->get_keyframe_metadata_map();
}

//-----------------------------------------------------------------------------

keyframe_data_graph
::keyframe_data_graph()
{

}

keyframe_data_graph
::~keyframe_data_graph()
{

}

keyframe_metadata_sptr 
keyframe_data_graph
::get_frame_metadata(frame_id_t frame) const
{
  return keyframe_metadata_sptr();
}

bool 
keyframe_data_graph
::set_frame_metadata(frame_id_t frame, keyframe_metadata_sptr metadata)
{
  return false;
}

bool
keyframe_data_graph
::remove_frame_metadata(frame_id_t frame)
{
  return false;
}

keyframe_data_map_const_sptr
keyframe_data_graph
::get_keyframe_metadata_map() const
{
  return keyframe_data_map_const_sptr();
}

keyframe_data_sptr 
keyframe_data_graph
::clone() const
{
  return keyframe_data_graph_sptr();
}



} // end namespace vital
} // end namespace kwiver