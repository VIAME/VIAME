/*ckwg +29
 * Copyright 2018-2021 by Kitware, Inc.
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

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <plugins/core/viame_core_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "add_timestamp_from_filename.h"
#include "auto_detect_transform.h"
#include "convert_head_tail_points.h"
#include "disparity_map_writer.h"
#include "empty_detector.h"
#include "read_detected_object_set_fishnet.h"
#include "read_detected_object_set_habcam.h"
#include "read_detected_object_set_oceaneyes.h"
#include "read_detected_object_set_viame_csv.h"
#include "write_detected_object_set_viame_csv.h"
#include "read_object_track_set_viame_csv.h"
#include "write_object_track_set_viame_csv.h"

namespace viame {

namespace {

static auto const module_name         = std::string{ "viame.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// ----------------------------------------------------------------------------
template <typename algorithm_t>
void register_algorithm( kwiver::vital::plugin_loader& vpm )
{
  using kvpf = kwiver::vital::plugin_factory;

  auto fact = vpm.ADD_ALGORITHM( algorithm_t::name, algorithm_t );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::description )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

extern "C"
VIAME_CORE_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< add_timestamp_from_filename >( vpm );
  register_algorithm< auto_detect_transform_io >( vpm );
  register_algorithm< convert_head_tail_points >( vpm );
  register_algorithm< disparity_map_writer >( vpm );
  register_algorithm< empty_detector >( vpm );
  register_algorithm< read_detected_object_set_fishnet >( vpm );
  register_algorithm< read_detected_object_set_habcam >( vpm );
  register_algorithm< read_detected_object_set_oceaneyes >( vpm );
  register_algorithm< read_detected_object_set_viame_csv >( vpm );
  register_algorithm< write_detected_object_set_viame_csv >( vpm );
  register_algorithm< read_object_track_set_viame_csv >( vpm );
  register_algorithm< write_object_track_set_viame_csv >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
