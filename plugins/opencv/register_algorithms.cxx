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

#include <plugins/opencv/viame_opencv_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "ocv_stereo_disparity_map.h"
#include "ocv_debayer_filter.h"
#include "ocv_random_hue_shift.h"
#include "ocv_image_enhancement.h"
#include "ocv_detect_calibration_targets.h"
#include "ocv_optimize_stereo_cameras.h"
#include "ocv_refiner_add_kps_from_mask.h"

#include "split_image_habcam.h"

namespace viame {

extern "C"
VIAME_OPENCV_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "viame.opencv" );

  if( reg.is_module_loaded() ) 
  {
    return;
  }

  reg.register_algorithm< ocv_stereo_disparity_map >();
  reg.register_algorithm< ocv_debayer_filter >();
  reg.register_algorithm< ocv_image_enhancement >();
  reg.register_algorithm< ocv_random_hue_shift >();
  reg.register_algorithm< ocv_detect_calibration_targets >();
  reg.register_algorithm< ocv_optimize_stereo_cameras >();
  reg.register_algorithm< ocv_refiner_add_kps_from_mask >();
  reg.register_algorithm< split_image_habcam >();

  reg.mark_module_as_loaded();
}

} // end namespace viame
