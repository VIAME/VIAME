// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

/// Compute the number of depth slices needed to properly sample the data
double compute_depth_sampling(world_space const& ws,
                              std::vector<vpgl_perspective_camera<double> > const& cameras);

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
