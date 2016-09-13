/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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


#ifndef KWIVER_ARROWS_FASTER_RCNN_DETECTOR_H_
#define KWIVER_ARROWS_FASTER_RCNN_DETECTOR_H_

#include <arrows/caffe/kwiver_algo_caffe_export.h>

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/object_labels.h>
#include <vital/algo/image_object_detector.h>
#include <vital/config/config_block.h>

#include <opencv2/core/core.hpp>

#include <caffe/blob.hpp>
#include <caffe/net.hpp>

#include <utility>

namespace kwiver {
namespace arrows {
namespace caffe {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class KWIVER_ALGO_CAFFE_EXPORT faster_rcnn_detector
  : public vital::algorithm_impl<faster_rcnn_detector, vital::algo::image_object_detector>
{
public:

  faster_rcnn_detector();
  faster_rcnn_detector( faster_rcnn_detector const& frd );

  virtual ~faster_rcnn_detector();

  virtual std::string impl_name() const { return "faster_rcnn_detector"; }

  virtual vital::config_block_sptr get_configuration() const;

  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual vital::detected_object_set_sptr detect( vital::image_container_sptr image_data) const;

private:

  class priv;
  const std::unique_ptr<priv> d;

};

}}}

#endif // KWIVER_ARROWS_FASTER_RCNN_DETECTOR_H_
