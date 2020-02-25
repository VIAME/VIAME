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
 * \brief Header file for tv_refine_search, minimizes a cost volume using total variation and
 *        exhaustive searching.
 */

#ifndef KWIVER_ARROWS_SUPER3D_TV_REFINE_SEARCH_H_
#define KWIVER_ARROWS_SUPER3D_TV_REFINE_SEARCH_H_

#include <vector>
#include <functional>
#include <memory>

#include <vil/vil_image_view.h>
#include <vnl/vnl_vector_fixed.h>

namespace kwiver {
namespace arrows {
namespace super3d {

class depth_refinement_monitor;

void
refine_depth(vil_image_view<double> &cost_volume,
             const vil_image_view<double> &g,
             vil_image_view<double> &d,
             unsigned int iterations,
             double theta0,
             double beta_end,
             double lambda,
             double epsilon,
             depth_refinement_monitor *drm = NULL);


class depth_refinement_monitor
{
public:

  struct update_data
  {
    vil_image_view<double> current_result;
    unsigned int num_iterations;
  };

  depth_refinement_monitor(std::function<bool (update_data)> callback, int interval) :
                           callback_(callback), interval_(interval)  {}

private:

  friend void refine_depth(vil_image_view<double> &cost_volume,
                           const vil_image_view<double> &g,
                           vil_image_view<double> &d,
                           unsigned int iterations,
                           double theta0,
                           double beta_end,
                           double lambda,
                           double epsilon,
                           depth_refinement_monitor *drm);

  std::function<bool (update_data)> callback_;
  int interval_;
};


//semi-implicit gradient ascent on q and descent on d
void
huber(vil_image_view<double> &q,
           vil_image_view<double> &d,
           const vil_image_view<double> &a,
           const vil_image_view<double> &g,
           double theta,
           double step,
           double epsilon);

void
min_search_bound(vil_image_view<double> &a,
  const vil_image_view<double> &d,
  const vil_image_view<double> &cost_volume,
  const vil_image_view<double> &cost_range,
  double theta,
  double lambda);

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver

#endif
