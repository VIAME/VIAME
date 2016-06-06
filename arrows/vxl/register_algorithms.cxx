/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * \brief Register VXL algorithms implementation
 */

#include "register_algorithms.h"

#include <arrows/algorithm_plugin_interface_macros.h>
#include <arrows/vxl/bundle_adjust.h>
#include <arrows/vxl/close_loops_homography_guided.h>
#include <arrows/vxl/estimate_essential_matrix.h>
#include <arrows/vxl/estimate_fundamental_matrix.h>
#include <arrows/vxl/estimate_homography.h>
#include <arrows/vxl/estimate_similarity_transform.h>
#include <arrows/vxl/image_io.h>
#include <arrows/vxl/optimize_cameras.h>
#include <arrows/vxl/triangulate_landmarks.h>
#include <arrows/vxl/match_features_constrained.h>


namespace kwiver {
namespace arrows {
namespace vxl {

/// Register VXL algorithm implementations with the given or global registrar
int register_algorithms( vital::registrar &reg )
{
  REGISTRATION_INIT( reg );

  REGISTER_TYPE( vxl::bundle_adjust );
  REGISTER_TYPE( vxl::close_loops_homography_guided );
  REGISTER_TYPE( vxl::estimate_essential_matrix );
  REGISTER_TYPE( vxl::estimate_fundamental_matrix );
  REGISTER_TYPE( vxl::estimate_homography );
  REGISTER_TYPE( vxl::estimate_similarity_transform );
  REGISTER_TYPE( vxl::image_io );
  REGISTER_TYPE( vxl::optimize_cameras );
  REGISTER_TYPE( vxl::triangulate_landmarks );
  REGISTER_TYPE( vxl::match_features_constrained );

  REGISTRATION_SUMMARY();
  return REGISTRATION_FAILURES();
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
