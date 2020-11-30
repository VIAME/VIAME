// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
