/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#ifndef VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
#define VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H

#include "viame_core_export.h"

#include <vital/algo/compute_track_descriptors.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <deque>
#include <map>
#include <vector>

namespace viame {

class VIAME_CORE_EXPORT average_track_descriptors
  : public kwiver::vital::algo::compute_track_descriptors
{
public:
  PLUGGABLE_IMPL(
    average_track_descriptors,
    "Track descriptor consolidation using simple averaging",
    PARAM_DEFAULT(
      rolling, bool,
      "When set, produce an output for each input as the rolling average "
      "of the last N descriptors, where N is the interval. When reset, "
      "produce an output only for the first input and then every Nth input "
      "thereafter for any given track.",
      false ),
    PARAM_DEFAULT(
      interval, unsigned int,
      "When the interval is N, every descriptor output (after the first N inputs) "
      "is based on the last N descriptors seen as input for the given track.",
      5 )
  )

  virtual ~average_track_descriptors() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::track_descriptor_set_sptr
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::object_track_set_sptr tracks );

  virtual kwiver::vital::track_descriptor_set_sptr flush();

private:
  void initialize() override;

  kwiver::vital::logger_handle_t m_logger;
  std::map< kwiver::vital::track_id_t, std::deque< std::vector< double > > > m_history;
};

} // end namespace viame

#endif // VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
