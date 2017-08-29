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

#ifndef KWIVER_ARROWS_BURNOUT_TRACK_DESCRIPTORS
#define KWIVER_ARROWS_BURNOUT_TRACK_DESCRIPTORS

#include <arrows/burnout/kwiver_algo_burnout_export.h>

#include <vital/vital_config.h>

#include <vital/algo/compute_track_descriptors.h>

namespace kwiver {
namespace arrows {
namespace burnout {

// ----------------------------------------------------------------
/**
 * @brief burnout_track_descriptors
 *
 */
class KWIVER_ALGO_BURNOUT_EXPORT burnout_track_descriptors
  : public vital::algorithm_impl< burnout_track_descriptors,
      vital::algo::compute_track_descriptors >
{
public:

  burnout_track_descriptors();
  virtual ~burnout_track_descriptors();

  virtual vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual kwiver::vital::track_descriptor_set_sptr
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::object_track_set_sptr tracks );

  virtual kwiver::vital::track_descriptor_set_sptr flush();

private:

  class priv;
  const std::unique_ptr<priv> d;
};

} } }

#endif /* KWIVER_ARROWS_BURNOUT_DETECTOR */
