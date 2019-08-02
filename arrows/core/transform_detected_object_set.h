/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#ifndef KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_
#define KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_filter.h>
#include <vital/io/camera_io.h>

namespace kwiver {
namespace arrows {
namespace core {

// ----------------------------------------------------------------
/**
 * @brief Transforms detections based on source and destination cameras.
 *
 */

class KWIVER_ALGO_CORE_EXPORT transform_detected_object_set
  : public vital::algorithm_impl<transform_detected_object_set, vital::algo::detected_object_filter>
{
public:
  PLUGIN_INFO( "transform_detected_object_set",
               "Transforms a detected object set based on source and destination cameras.\n\n" )

  transform_detected_object_set();
  virtual ~transform_detected_object_set() = default;

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual vital::detected_object_set_sptr filter( const vital::detected_object_set_sptr input_set) const;

private:
  std::string src_camera_krtd_file_name;
  std::string dest_camera_krtd_file_name;

  kwiver::vital::camera_perspective_sptr src_camera;
  kwiver::vital::camera_perspective_sptr dest_camera;

  virtual vital::bounding_box<double> transform_bounding_box(vital::bounding_box<double>& bbox) const;
};

}}} //End namespace


#endif // KWIVER_ARROWS_TRANSFORM_DETECTED_OBJECT_SET_H_
