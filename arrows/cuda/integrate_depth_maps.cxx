/*ckwg +29
* Copyright 2016, Kitware SAS, Copyright 2018 by Kitware, Inc.
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
* \brief Source file for compute_depth, driver for depth from an image sequence
*/

#include <arrows/cuda/integrate_depth_maps.h>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace kwiver::vital;

void cuda_initalize(int h_gridDims[3], double h_gridOrig[3], double h_gridSpacing[3],
                    double h_rayPThick, double h_rayPRho, double h_rayPEta, double h_rayPDelta);

void launch_depth_kernel(double * d_depth, int depthmap_dims[2], double d_K[16], double d_RT[16], double* output);

namespace kwiver {
namespace arrows {
namespace cuda {

/// Private implementation class
class integrate_depth_maps::priv
{
public:
  /// Constructor
  priv()
    : ray_potential_thickness(1.65),
      ray_potential_rho(1.0),
      ray_potential_eta(0.03),
      ray_potential_delta(16.5),
      grid_spacing {1.0, 1.0, 1.0},
      m_logger(vital::get_logger("arrows.cuda.integrate_depth_maps"))
  {
  }

  double ray_potential_rho;
  double ray_potential_thickness;
  double ray_potential_eta;
  double ray_potential_delta;

  int grid_dims[3];
  double grid_spacing[3];

  /// Logger handle
  vital::logger_handle_t m_logger;
};

//*****************************************************************************

/// Constructor
integrate_depth_maps::integrate_depth_maps()
  : d_(new priv)
{
}

//*****************************************************************************

/// Destructor
integrate_depth_maps::~integrate_depth_maps()
{
}

//*****************************************************************************

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
integrate_depth_maps::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::integrate_depth_maps::get_configuration();

  config->set_value("ray_potential_thickness", d_->ray_potential_thickness, "ray potential thickness");
  config->set_value("ray_potential_rho", d_->ray_potential_rho, "ray potential rho");
  config->set_value("ray_potential_eta", d_->ray_potential_eta,
                     "0 < Eta < 1 : will be applied as a percentage of rho ");
  config->set_value("ray_potential_delta", d_->ray_potential_delta,
                    "delta has to be superior to Thick ");

  std::ostringstream stream;
  stream << d_->grid_spacing[0] << " " << d_->grid_spacing[1] << " " << d_->grid_spacing[2];
  config->set_value("grid_spacing", stream.str(), "spacing for each dimension of the grid");

  return config;
}

//*****************************************************************************

/// Set this algorithm's properties via a config block
void
integrate_depth_maps::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that
  // assumed values are present. An alternative is to check for key
  // presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->ray_potential_rho = config->get_value<double>("ray_potential_rho", d_->ray_potential_rho);
  d_->ray_potential_thickness = config->get_value<double>("ray_potential_thickness", d_->ray_potential_thickness);
  d_->ray_potential_eta = config->get_value<double>("ray_potential_eta", d_->ray_potential_eta);
  d_->ray_potential_delta = config->get_value<double>("ray_potential_delta", d_->ray_potential_delta);

  std::ostringstream ostream;
  ostream << d_->grid_spacing[0] << " " << d_->grid_spacing[1] << " " << d_->grid_spacing[2];
  std::string spacing = config->get_value<std::string>("grid_spacing", ostream.str());
  std::istringstream istream(spacing);
  istream >> d_->grid_spacing[0] >> d_->grid_spacing[1] >> d_->grid_spacing[2];
}

//*****************************************************************************

/// Check that the algorithm's currently configuration is valid
bool
integrate_depth_maps::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

//*****************************************************************************

double *copy_depth_map_to_gpu(kwiver::vital::image_container_sptr h_depth)
{
  size_t size = h_depth->height() * h_depth->width();
  double* temp = new double[size];

  //copy to cuda format
  kwiver::vital::image img = h_depth->get_image();
  for (unsigned int i = 0; i < h_depth->width(); i++)
  {
    for (unsigned int j = 0; j < h_depth->height(); j++)
    {
      temp[i*h_depth->height() + j] = img.at<double>(i, j);
    }
  }

  double *d_depth;
  cudaMalloc((void**)&d_depth, size * sizeof(double));
  cudaMemcpy(d_depth, temp, size * sizeof(double), cudaMemcpyHostToDevice);
  delete [] temp;

  return d_depth;
}

//*****************************************************************************

double *init_volume_on_gpu(unsigned int vsize)
{
  double *output;
  cudaMalloc((void**)&output, vsize * sizeof(double));
  cudaMemset(output, 0, vsize * sizeof(double));
  return output;
}

//*****************************************************************************

void copy_camera_to_gpu(kwiver::vital::camera_perspective_sptr camera, double* d_K, double *d_RT)
{
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> K4x4;
  K4x4.setIdentity();
  matrix_3x3d K(camera->intrinsics()->as_matrix());
  K4x4.block< 3, 3 >(0, 0) = K;

  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> RT;
  RT.setIdentity();
  matrix_3x3d R(camera->rotation().matrix());
  vector_3d t(camera->translation());
  RT.block< 3, 3 >(0, 0) = R;
  RT.block< 3, 1 >(0, 3) = t;

  cudaMemcpy(d_K, K4x4.data(), 16 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_RT, RT.data(), 16 * sizeof(double), cudaMemcpyHostToDevice);
}


//*****************************************************************************

void
integrate_depth_maps::integrate(
  vector_3d const& minpt_bound, vector_3d const& maxpt_bound,
  std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
  std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
  kwiver::vital::image_container_sptr& volume) const
{
  vector_3d diff = maxpt_bound - minpt_bound;
  vector_3d orig = minpt_bound;

  for (int i = 0; i < 3; i++)
  {
    d_->grid_dims[i] = static_cast<int>((diff[i] / d_->grid_spacing[i]));
  }

  LOG_DEBUG( logger(), "grid: " << d_->grid_dims[0]
                       << " "   << d_->grid_dims[1]
                       << " "   << d_->grid_dims[2] );

  LOG_INFO( logger(), "initialize" );
  cuda_initalize(d_->grid_dims, orig.data(), d_->grid_spacing,
    d_->ray_potential_thickness, d_->ray_potential_rho, d_->ray_potential_eta, d_->ray_potential_delta);
  const int vsize = d_->grid_dims[0] * d_->grid_dims[1] * d_->grid_dims[2];

  double *d_volume = init_volume_on_gpu(vsize);
  double *d_K, *d_RT;

  cudaMalloc((void**)&d_K, 16 * sizeof(double));
  cudaMalloc((void**)&d_RT, 16 * sizeof(double));

  for (int i = 0; i < depth_maps.size(); i++)
  {
    int depthmap_dims[2];
    depthmap_dims[0] = static_cast<int>(depth_maps[i]->width());
    depthmap_dims[1] = static_cast<int>(depth_maps[i]->height());
    double *d_depth = copy_depth_map_to_gpu(depth_maps[i]);
    copy_camera_to_gpu(cameras[i], d_K, d_RT);

    // run code on device
    LOG_INFO( logger(), "depth map " << i );
    launch_depth_kernel(d_depth, depthmap_dims, d_K, d_RT, d_volume);
    cudaFree(d_depth);
  }

  // Transfer data from device to host
  double *h_volume = new double[vsize];
  cudaMemcpy(h_volume, d_volume, vsize * sizeof(double), cudaMemcpyDeviceToHost);

  volume = std::shared_ptr<image_container>(new simple_image_container(
    image(h_volume, d_->grid_dims[0], d_->grid_dims[1], d_->grid_dims[2],
      1, d_->grid_dims[0], d_->grid_dims[0] * d_->grid_dims[1],
      image_pixel_traits(kwiver::vital::image_pixel_traits::FLOAT, 8))));

  cudaFree(d_volume);
  cudaFree(d_K);
  cudaFree(d_RT);
}

} // end namespace cuda
} // end namespace arrows
} // end namespace kwiver
