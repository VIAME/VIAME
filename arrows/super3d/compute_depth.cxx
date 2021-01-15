// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Source file for compute_depth, driver for depth from an image sequence
*/

#include <arrows/super3d/compute_depth.h>

#include <arrows/vxl/camera.h>
#include <arrows/vxl/image_container.h>
#include <vil/algo/vil_threshold.h>
#include <vil/vil_convert.h>
#include <vil/vil_crop.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>
#include <vital/types/bounding_box.h>
#include <vital/vital_config.h>

#include <sstream>
#include <memory>
#include <functional>

#include "cost_volume.h"
#include "tv_refine_search.h"
#include "world_angled_frustum.h"
#include "util.h"

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace super3d {

/// Private implementation class
class compute_depth::priv
{
public:
  /// Constructor
  priv()
    : theta0(1.0),
      theta_end(0.001),
      lambda(0.65),
      gw_alpha(20),
      epsilon(0.01),
      iterations(2000),
      world_plane_normal(0.0, 0.0, 1.0),
      depth_sample_rate(0.5),
      callback_interval(-1),    //default is no callback
      uncertainty_in_callback(false),
      callback(NULL),
      m_logger(vital::get_logger("arrows.super3d.compute_depth"))
  {
  }

  bool iterative_update_callback(depth_refinement_monitor::update_data data);
  bool cost_volume_update_callback(unsigned int slice_num);

  std::unique_ptr<world_space> compute_world_space_roi(vpgl_perspective_camera<double> &cam,
                                                       vil_image_view<double> &frame,
                                                       double d_min, double d_max,
                                                       vital::bounding_box<int> const& roi);

  double theta0;
  double theta_end;
  double lambda;
  double gw_alpha;
  double epsilon;
  unsigned int iterations;
  vnl_double_3 world_plane_normal;
  double depth_sample_rate;
  int callback_interval;
  bool uncertainty_in_callback;

  double depth_min, depth_max;
  unsigned int num_slices;

  vpgl_perspective_camera<double> ref_cam;

  vil_image_view<double> cost_volume;

  compute_depth::callback_t callback;

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
  config->set_value("iterations", d_->iterations,
                    "Number of iterations to run optimizer");
  config->set_value("theta0", d_->theta0,
                    "Begin value of quadratic relaxation term");
  config->set_value("theta_end", d_->theta_end,
                    "End value of quadratic relaxation term");
  config->set_value("lambda", d_->lambda,
                    "Weight of the data term");
  config->set_value("gw_alpha", d_->gw_alpha,
                    "gradient weighting term");
  config->set_value("epsilon", d_->epsilon,
                    "Huber norm term, trade off between L1 and L2 norms");
  config->set_value("world_plane_normal", "0 0 1",
                    "up direction in world space");
  config->set_value("callback_interval", d_->callback_interval,
                    "number of iterations between updates (-1 turns off updates)");
  config->set_value("uncertainty_in_callback", d_->uncertainty_in_callback,
                    "If true, compute the uncertainty in each callback for a "
                    "live preview at additional computational cost. "
                    "Otherwise, uncertainty is only computed at the end.");
  config->set_value("depth_sample_rate", d_->depth_sample_rate,
                    "Specifies the maximum sampling rate, in pixels, of the "
                    "depth steps projected into support views.  This rate "
                    "determines the number of depth slices in the cost "
                    "volume.  Smaller values create more depth slices.");

  return config;
}

//*****************************************************************************

/// Set this algorithm's properties via a config block
void
compute_depth::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that
  // assumed values are present. An alternative is to check for key
  // presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->iterations = config->get_value<unsigned int>("iterations", d_->iterations);
  d_->theta0 = config->get_value<double>("theta0", d_->theta0);
  d_->theta_end = config->get_value<double>("theta_end", d_->theta_end);
  d_->lambda = config->get_value<double>("lambda", d_->lambda);
  d_->gw_alpha = config->get_value<double>("gw_alpha", d_->gw_alpha);
  d_->epsilon = config->get_value<double>("epsilon", d_->epsilon);
  d_->callback_interval = config->get_value<double>("callback_interval",
                                                    d_->callback_interval);
  d_->uncertainty_in_callback =
    config->get_value<bool>("uncertainty_in_callback",
                            d_->uncertainty_in_callback);
  d_->depth_sample_rate = config->get_value<double>("depth_sample_rate",
                                                    d_->depth_sample_rate);

  std::istringstream ss(config->get_value<std::string>("world_plane_normal",
                                                       "0 0 1"));
  ss >> d_->world_plane_normal;
  d_->world_plane_normal.normalize();
}

//*****************************************************************************

/// Check that the algorithm's currently configuration is valid
bool
compute_depth::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

//*****************************************************************************

//Will crop the reference camera and frame passed in
std::unique_ptr<world_space>
compute_depth::priv
::compute_world_space_roi(vpgl_perspective_camera<double> &cam,
                          vil_image_view<double> &frame,
                          double d_min, double d_max,
                          vital::bounding_box<int> const& roi)
{
  frame = vil_crop(frame, roi.min_x(), roi.width(), roi.min_y(), roi.height());
  cam = crop_camera(cam, roi.min_x(), roi.min_y());

  return std::unique_ptr<world_space>(new world_angled_frustum(cam, world_plane_normal,
                                                                d_min, d_max, roi.width(), roi.height()));
}

//*****************************************************************************

vil_image_view<double>
compute_uncertainty(vil_image_view<double> const& height_map,
                    vil_image_view<double> const& cost_volume)
{
  const int S = static_cast<int>(cost_volume.nplanes());
  vil_image_view<double> uncertainty(height_map.ni(), height_map.nj(), 1);
  // This scale is 1/(2*sigma) converted from [0,255] to [0,1]
  const double cost_scale = 255.0 / (2.0 * 5.0);

#pragma omp parallel for
  for (int64_t j = 0; j < cost_volume.nj(); j++)
  {
    for (unsigned int i = 0; i < cost_volume.ni(); i++)
    {
      double const& dij = height_map(i, j);
      double sum_w = 0.0;
      double var = 0.0;
      for (unsigned int k = 0; k < cost_volume.nplanes(); k++)
      {
        const double d_k = (static_cast<double>(k) + 0.5) / S;
        const double diff = d_k - dij;
        const double w = std::exp(-cost_volume(i, j, k) * cost_scale);
        sum_w += w;
        var += w * diff * diff;
      }
      uncertainty(i, j) = std::sqrt(var / sum_w);
    }
  }
  return uncertainty;
}

//*****************************************************************************

//Compute the depth and return the uncertainty by reference
image_container_sptr
compute_depth
::compute(std::vector<kwiver::vital::image_container_sptr> const& frames_in,
          std::vector<kwiver::vital::camera_perspective_sptr> const& cameras_in,
          double depth_min, double depth_max,
          unsigned int ref_frame,
          vital::bounding_box<int> const& roi,
          kwiver::vital::image_container_sptr& depth_uncertainty,
          std::vector<kwiver::vital::image_container_sptr> const& masks_in) const
{

  //convert frames
  std::vector<vil_image_view<double> > frames(frames_in.size());
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < static_cast< int >(frames.size()); i++)
  {
    vil_image_view<vxl_byte> img =
      vxl::image_container::vital_to_vxl(frames_in[i]->get_image());
    vil_convert_planes_to_grey(img, frames[i]);
    vil_math_scale_values(frames[i], 1.0 / 255.0);
  }

  d_->depth_min = depth_min;
  d_->depth_max = depth_max;

  //convert optional mask images
  std::vector<vil_image_view<bool> > masks;
  vil_image_view<bool> *ref_mask = NULL;
  if (!masks_in.empty())
  {
    masks.resize(masks_in.size());
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < static_cast< int >(masks.size()); i++)
    {
      if (!masks_in[i])
      {
        continue;
      }
      auto vxl_mask = vxl::image_container::vital_to_vxl(masks_in[i]->get_image());
      if (!vxl_mask)
      {
        continue;
      }
      if (vxl_mask->pixel_format() == VIL_PIXEL_FORMAT_BOOL)
      {
        masks[i] = vxl_mask;
      }
      else if (vxl_mask->pixel_format() == VIL_PIXEL_FORMAT_BYTE)
      {
        vil_threshold_above<vxl_byte>(vxl_mask, masks[i], 128);
      }
      else
      {
        // unsupported pixel format
        continue;
      }
      // ensure that this is a single channel image
      // take only the first channel
      masks[i] = vil_plane(masks[i], 0);
    }
    ref_mask = &masks[ref_frame];
  }

  //convert cameras
  std::vector<vpgl_perspective_camera<double> > cameras(cameras_in.size());
  for (unsigned int i = 0; i < cameras.size(); i++) {
    vxl::vital_to_vpgl_camera<double>(*cameras_in[i], cameras[i]);
  }

  std::unique_ptr<world_space> ws = d_->compute_world_space_roi(cameras[ref_frame], frames[ref_frame], depth_min, depth_max, roi);

  double depth_sampling = compute_depth_sampling(*ws, cameras) /
                            d_->depth_sample_rate;
  d_->num_slices = static_cast<unsigned int>(depth_sampling);

  d_->ref_cam = cameras[ref_frame];
  vil_image_view<double> g;

  cost_volume_callback_t cv_callback = std::bind1st(std::mem_fun(
    &compute_depth::priv::cost_volume_update_callback), this->d_.get());
  if (!compute_world_cost_volume(frames, cameras, ws.get(), ref_frame,
                                 d_->num_slices, d_->cost_volume,
                                 cv_callback, masks))
  {
    // user terminated processing early through the callback
    return nullptr;
  }

  LOG_DEBUG(d_->m_logger, "Computing g weighting");
  compute_g(frames[ref_frame], g, d_->gw_alpha, 1.0, ref_mask);

  LOG_DEBUG(d_->m_logger, "Refining Depth");
  vil_image_view<double> height_map(d_->cost_volume.ni(), d_->cost_volume.nj(), 1);

  if (!d_->callback)
  {
    refine_depth(d_->cost_volume, g, height_map, d_->iterations,
                 d_->theta0, d_->theta_end, d_->lambda, d_->epsilon);
  }
  else
  {
    std::function<bool (depth_refinement_monitor::update_data)> f;
    f = std::bind1st(std::mem_fun(&compute_depth::priv::iterative_update_callback),
                     this->d_.get());
    depth_refinement_monitor *drm =
      new depth_refinement_monitor(f, d_->callback_interval);
    refine_depth(d_->cost_volume, g, height_map, d_->iterations,
                 d_->theta0, d_->theta_end, d_->lambda, d_->epsilon, drm);
    delete drm;
  }

  auto uncertainty = compute_uncertainty(height_map, d_->cost_volume);

  // map depth from normalized range back into true depth
  double scale = depth_max - depth_min;
  vil_math_scale_and_offset_values(height_map, scale, depth_min);
  vil_math_scale_values(uncertainty, scale);

  vil_image_view<double> depth;
  height_map_to_depth_map(d_->ref_cam, height_map, depth, uncertainty);

  // Setting the value by reference
  depth_uncertainty = std::make_shared<vxl::image_container>(uncertainty);

  return vital::image_container_sptr(new vxl::image_container(depth));
}

//*****************************************************************************

void compute_depth::set_callback(callback_t cb)
{
  kwiver::vital::algo::compute_depth::set_callback(cb);
  d_->callback = cb;
}

//*****************************************************************************

//Bridge from super3d monitor to vital image
bool
compute_depth::priv
::iterative_update_callback(depth_refinement_monitor::update_data data)
{
  if (this->callback)
  {
    image_container_sptr result = nullptr;
    image_container_sptr result_u = nullptr;
    if (data.current_result)
    {
      vil_image_view<double> depth;
      double depth_scale = this->depth_max - this->depth_min;
      if (this->uncertainty_in_callback)
      {
        auto uncertainty = compute_uncertainty(data.current_result,
                                               this->cost_volume);
        vil_math_scale_values(uncertainty, depth_scale);
        vil_math_scale_and_offset_values(data.current_result,
                                         depth_scale, this->depth_min);
        height_map_to_depth_map(this->ref_cam, data.current_result,
                                depth, uncertainty);
        result_u = std::make_shared<vxl::image_container>(
          vxl::image_container::vxl_to_vital(uncertainty));
      }
      else
      {
        vil_math_scale_and_offset_values(data.current_result,
                                         depth_scale, this->depth_min);
        height_map_to_depth_map(this->ref_cam, data.current_result,
                                depth);
      }
      result = std::make_shared<vxl::image_container>(
                 vxl::image_container::vxl_to_vital(depth));
    }
    unsigned percent_complete = 50 + (50 * data.num_iterations)
                                     / this->iterations;
    std::stringstream ss;
    ss << "Depth refinement iteration " << data.num_iterations
       << " of " << this->iterations;

    return this->callback(result, ss.str(), percent_complete, result_u);
  }
  return true;
}

//Bridge from super3d cost volume computation  monitor
bool
compute_depth::priv
::cost_volume_update_callback(unsigned int slice_num)
{
  if (this->callback)
  {
    unsigned percent_complete = (50 * slice_num) / this->num_slices;
    std::stringstream ss;
    ss << "Computing cost volume slice " << slice_num << " of " << this->num_slices;
    return this->callback(nullptr, ss.str(), percent_complete, nullptr);
  }
  return true;
}

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
