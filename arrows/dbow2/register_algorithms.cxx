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
 * \brief OpenCV algorithm registration implementation
 */

#include <arrows/dbow2/kwiver_algo_dbow2_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_NONFREE
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include <arrows/dbow2/match_descriptor_sets.h>

namespace kwiver {
namespace arrows {
namespace dbow2 {

extern "C"
KWIVER_ALGO_DBOW2_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.dbow2" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#if defined(HAVE_OPENCV_NONFREE)
  cv::initModule_nonfree();
#endif

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "dbow2", kwiver::arrows::dbow2::match_descriptor_sets );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Use DBoW2 for bag of words matching of descriptor sets. "
                    "This is currently limited to OpenCV ORB descriptors.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  vpm.mark_module_as_loaded( module_name );
}

} // end namespace dbow2
} // end namespace arrows
} // end namespace kwiver
