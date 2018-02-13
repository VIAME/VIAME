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
* \brief Source file for compute_depth, driver for depth estimation from an image sequence
*/

#include <arrows/depth/compute_depth.h>
#include <arrows/vxl/image_container.h>
#include <vital/types/landmark.h>
#include <arrows/vxl/camera.h>
#include <vnl/vnl_double_3.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vpgl/vpgl_perspective_camera.h>

#include <sstream>
#include <memory>

#include "cost_volume.h"
#include "tv_refine_search.h"
#include "world_angled_frustum.h"

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace depth {

/// Private implementation class
class compute_depth::priv
{
public:
  /// Constructor
  priv()
    : S(30),
      theta0(1.0),
      theta_end(0.001),
      lambda(0.65),
      gw_alpha(20),
      epsilon(0.01),
      iterations(2000),
      world_plane_normal(0.0, 0.0, 1.0),
      m_logger(vital::get_logger("arrows.depth.compute_depth"))
  {
  }

  unsigned int S;
  double theta0;
  double theta_end;
  double lambda;
  double gw_alpha;
  double epsilon;
  unsigned int iterations;
  vnl_double_3 world_plane_normal;

  /// Logger handle
  vital::logger_handle_t m_logger;
};

//*****************************************************************************

/// Constructor
compute_depth::compute_depth()
  : d_(new priv)
{
}

//*****************************************************************************

/// Destructor
compute_depth::~compute_depth()
{
}

//*****************************************************************************

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
compute_depth::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::compute_depth::get_configuration();
  config->set_value("iterations", d_->iterations, "Number of iterations to run optimizer");
  config->set_value("theta0", d_->theta0, "Begin value of quadratic relaxation term");
  config->set_value("theta_end", d_->theta_end, "End value of quadratic relaxation term");
  config->set_value("lambda", d_->lambda, "Weight of the data term");
  config->set_value("gw_alpha", d_->gw_alpha, "gradient weighting term");
  config->set_value("epsilon", d_->epsilon, "Huber norm term, trade off between L1 and L2 norms");
  config->set_value("world_plane_normal", "0 0 1", "up direction in world space");
  return config;
}

//*****************************************************************************

/// Set this algorithm's properties via a config block
void
compute_depth::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->iterations = config->get_value<unsigned int>("iterations", d_->iterations);
  d_->theta0 = config->get_value<double>("theta0", d_->theta0);
  d_->theta_end = config->get_value<double>("theta_end", d_->theta_end);
  d_->lambda = config->get_value<double>("lambda", d_->lambda);
  d_->gw_alpha = config->get_value<double>("gw_alpha", d_->gw_alpha);
  d_->epsilon = config->get_value<double>("epsilon", d_->epsilon);

  std::istringstream ss(config->get_value<std::string>("world_plane_normal", "0 0 1"));
  ss >> d_->world_plane_normal;
  d_->world_plane_normal.normalize();  
}

//*****************************************************************************

/// Check that the algorithm's currently configuration is valid
bool
compute_depth::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

//*****************************************************************************

image_container_sptr
compute_depth::compute(const std::vector<image_container_sptr> &frames_in,
                       const std::vector<camera_sptr> &cameras_in,
                       const std::vector<landmark_sptr> &landmarks_in,
                       unsigned int ref_frame,
                       std::vector<image_container_sptr> *masks_in) const
{
  //convert frames
  std::vector<vil_image_view<double> > frames(frames_in.size());
  for (unsigned int i = 0; i < frames.size(); i++) {
    frames[i] = vxl::image_container::vital_to_vxl(frames_in[i]->get_image());
  }

  //convert optional mask images
  std::vector<vil_image_view<double> > *masks = NULL;
  vil_image_view<double> *ref_mask = NULL;
  if (masks_in) 
  {
    masks = new std::vector<vil_image_view<double> >(masks_in->size());
    for (unsigned int i = 0; i < masks->size(); i++) {
      (*masks)[i] = vxl::image_container::vital_to_vxl((*masks_in)[i]->get_image());
    }
    ref_mask = &(*masks)[ref_frame];
  }

  //convert cameras
  std::vector<vpgl_perspective_camera<double> > cameras(cameras_in.size());
  for (unsigned int i = 0; i < cameras.size(); i++) {
    vxl::vital_to_vpgl_camera<double>(*cameras_in[i], cameras[i]);
  }

  //convert landmarks
  std::vector<vnl_double_3> landmarks(landmarks_in.size());
  for (unsigned int i = 0; i < landmarks.size(); i++) {
    landmarks[i] = vnl_vector_fixed<double, 3>(landmarks_in[i]->loc().data());
  }
  
  vpgl_perspective_camera<double> ref_cam = cameras[ref_frame];

  int ni = frames[ref_frame].ni(), nj = frames[ref_frame].nj();
  double depth_min, depth_max;

  
  std::vector<vnl_double_3> visible_landmarks =
    filter_visible_landmarks(cameras[ref_frame], 0, ni, 0, nj, landmarks);
  compute_offset_range(visible_landmarks, d_->world_plane_normal, depth_min, depth_max, 0.1, 0.5);
  world_space *ws = new world_angled_frustum(cameras[ref_frame], d_->world_plane_normal, depth_min, depth_max, ni, nj);
  
  vil_image_view<double> g;
  vil_image_view<double> cost_volume;

  compute_world_cost_volume(frames, cameras, ws, ref_frame, d_->S, cost_volume, masks);
  compute_g(frames[ref_frame], g, d_->gw_alpha, 1.0, ref_mask);

  std::cout << "Refining Depth. ..\n";
  vil_image_view<double> depth(cost_volume.ni(), cost_volume.nj(), 1);
  
  refine_depth(cost_volume, g, depth, d_->iterations, d_->theta0, d_->theta_end, d_->lambda, d_->epsilon);

  // map depth from normalized range back into true depth
  double depth_scale = depth_max - depth_min;
  vil_math_scale_and_offset_values(depth, depth_scale, depth_min);

  delete ws;
  if (masks) delete masks;

  return vital::image_container_sptr(new vxl::image_container(depth));
}

//*****************************************************************************

} // end namespace depth
} // end namespace arrows
} // end namespace kwiver