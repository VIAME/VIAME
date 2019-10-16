/*ckwg +29
 * Copyright 2012-2019 by Kitware, Inc.
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
 * \brief Header file for cost_volume
 */

#ifndef KWIVER_ARROWS_SUPER3D_COST_VOLUME_H_
#define KWIVER_ARROWS_SUPER3D_COST_VOLUME_H_

#include <vector>
#include <functional>
#include <vil/vil_image_view.h>
#include <vpgl/vpgl_perspective_camera.h>

#include "world_space.h"

namespace kwiver {
namespace arrows {
namespace super3d {

/// Typedef for the callback function signature
/**
 * The cost volume callback is called with current depth slice
 * to report progress on computing the cost volume.
 * If the callback returns false, the cost volume computation
 * will abort early.  If true it will continue.
 */
typedef std::function<bool(unsigned int)> cost_volume_callback_t;

bool
compute_world_cost_volume(const std::vector<vil_image_view<double> > &frames,
                          const std::vector<vpgl_perspective_camera<double> > &cameras,
                          world_space *ws,
                          unsigned int ref_frame,
                          unsigned int S,
                          vil_image_view<double> &cost_volume,
                          cost_volume_callback_t callback = nullptr,
                          const std::vector<vil_image_view<bool> > &masks =
                            std::vector<vil_image_view<bool> >());

//Compute gradient weights
void compute_g(const vil_image_view<double> &ref_img,
         vil_image_view<double> &g,
         double alpha,
         double beta,
         vil_image_view<bool> *mask = NULL);

void save_cost_volume(const vil_image_view<double> &cost_volume,
                      const vil_image_view<double> &g_weight,
                      const char *file_name);

void load_cost_volume(vil_image_view<double> &cost_volume,
                      vil_image_view<double> &g_weight,
                      const char *file_name);

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver


#endif
