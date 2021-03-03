// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Source file for compute_depth, driver for depth from an image sequence
*/

#include <arrows/cuda/integrate_depth_maps.h>
#include <arrows/cuda/cuda_error_check.h>
#include <arrows/cuda/cuda_memory.h>
#include <arrows/core/depth_utils.h>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace kwiver::vital;

void cuda_initalize(int h_gridDims[3], double h_gridOrig[3],
                    double h_gridSpacing[3], double h_rayPThick,
                    double h_rayPRho, double h_rayPEta, double h_rayPEpsilon,
                    double h_rayPDelta);

void launch_depth_kernel(double * d_depth, double * d_weight, int depthmap_dims[2],
                         double d_K[16], double d_RT[16], double* output,
                         unsigned max_voxels_per_launch);

namespace kwiver {
namespace arrows {
namespace cuda {

/// Private implementation class
class integrate_depth_maps::priv
{
public:
  // Constructor
  priv()
    : ray_potential_rho(1.0),
      ray_potential_thickness(20.0),
      ray_potential_eta(1.0),
      ray_potential_epsilon(0.01),
      ray_potential_delta(200.0),
      grid_spacing {1.0, 1.0, 1.0},
      voxel_spacing_factor(1.0),
      max_voxels_per_launch(20000000),
      m_logger(vital::get_logger("arrows.cuda.integrate_depth_maps"))
  {
  }

  double ray_potential_rho;
  double ray_potential_thickness;
  double ray_potential_eta;
  double ray_potential_epsilon;
  double ray_potential_delta;

  int grid_dims[3];

  // Actual spacing is computed as
  //   voxel_scale_factor * pixel_to_world_scale * grid_spacing
  // relative spacings per dimension
  double grid_spacing[3];

  // multiplier on all dimensions of grid spacing
  double voxel_spacing_factor;

  // Maximum number of voxels to process in a single kernel launch
  unsigned max_voxels_per_launch;

  // Logger handle
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
  auto config = vital::algo::integrate_depth_maps::get_configuration();

  config->set_value("ray_potential_thickness", d_->ray_potential_thickness,
                    "Distance that the TSDF covers sloping from Rho to zero. "
                    "Units are in voxels.");
  config->set_value("ray_potential_rho", d_->ray_potential_rho,
                    "Maximum magnitude of the TDSF");
  config->set_value("ray_potential_eta", d_->ray_potential_eta,
                    "Fraction of rho to use for free space constraint. "
                    "Requires 0 <= Eta <= 1.");
  config->set_value("ray_potential_epsilon", d_->ray_potential_epsilon,
                    "Fraction of rho to use in occluded space. "
                    "Requires 0 <= Epsilon <= 1.");
  config->set_value("ray_potential_delta", d_->ray_potential_delta,
                    "Distance from the surface before the TSDF is truncate. "
                    "Units are in voxels");
  config->set_value("voxel_spacing_factor", d_->voxel_spacing_factor,
                    "Multiplier on voxel spacing.  Set to 1.0 for voxel "
                    "sizes that project to 1 pixel on average.");
  config->set_value("max_voxels_per_launch", d_->max_voxels_per_launch,
                    "Maximum number of voxels to process in a single kernel "
                    "launch.  Processing too much data at once on the GPU "
                    "can cause the GPU to time out.  Set to zero for "
                    "unlimited.");

  std::ostringstream stream;
  stream << d_->grid_spacing[0] << " "
         << d_->grid_spacing[1] << " "
         << d_->grid_spacing[2];
  config->set_value("grid_spacing", stream.str(),
                    "Relative spacing for each dimension of the grid");

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

  d_->ray_potential_rho =
    config->get_value<double>("ray_potential_rho", d_->ray_potential_rho);
  d_->ray_potential_thickness =
    config->get_value<double>("ray_potential_thickness",
                              d_->ray_potential_thickness);
  d_->ray_potential_eta =
    config->get_value<double>("ray_potential_eta", d_->ray_potential_eta);
  d_->ray_potential_epsilon =
    config->get_value<double>("ray_potential_epsilon", d_->ray_potential_epsilon);
  d_->ray_potential_delta =
    config->get_value<double>("ray_potential_delta", d_->ray_potential_delta);
  d_->voxel_spacing_factor =
    config->get_value<double>("voxel_spacing_factor", d_->voxel_spacing_factor);
  d_->max_voxels_per_launch =
    config->get_value<unsigned>("max_voxels_per_launch",
                                d_->max_voxels_per_launch);

  std::ostringstream ostream;
  ostream << d_->grid_spacing[0] << " "
          << d_->grid_spacing[1] << " "
          << d_->grid_spacing[2];
  std::string spacing =
    config->get_value<std::string>("grid_spacing", ostream.str());
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

cuda_ptr<double>
copy_img_to_gpu(kwiver::vital::image_container_sptr h_img)
{
  size_t size = h_img->height() * h_img->width();
  std::unique_ptr<double[]> temp(new double[size]);

  //copy to cuda format
  kwiver::vital::image img = h_img->get_image();
  for (unsigned int j = 0; j < h_img->height(); j++)
  {
    for (unsigned int i = 0; i < h_img->width(); i++)
    {
      temp[j*h_img->width() + i] = img.at<double>(i, j);
    }
  }

  auto d_img = make_cuda_mem<double>(size);
  CudaErrorCheck(cudaMemcpy(d_img.get(), temp.get(), size * sizeof(double),
                            cudaMemcpyHostToDevice));

  return d_img;
}

//*****************************************************************************

cuda_ptr<double> init_volume_on_gpu(size_t vsize)
{
  auto output = make_cuda_mem<double>(vsize);
  CudaErrorCheck(cudaMemset(output.get(), 0, vsize * sizeof(double)));
  return output;
}

//*****************************************************************************

void copy_camera_to_gpu(kwiver::vital::camera_perspective_sptr camera,
                        double* d_K, double *d_RT)
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

  CudaErrorCheck(cudaMemcpy(d_K, K4x4.data(), 16 * sizeof(double),
                            cudaMemcpyHostToDevice));
  CudaErrorCheck(cudaMemcpy(d_RT, RT.data(), 16 * sizeof(double),
                            cudaMemcpyHostToDevice));
}

//*****************************************************************************

void
integrate_depth_maps::integrate(
  vector_3d const& minpt_bound,
  vector_3d const& maxpt_bound,
  std::vector<image_container_sptr> const& depth_maps,
  std::vector<image_container_sptr> const& weight_maps,
  std::vector<camera_perspective_sptr> const& cameras,
  image_container_sptr& volume,
  vector_3d &spacing) const
{
  double pixel_to_world_scale;
  pixel_to_world_scale =
    kwiver::arrows::core::
      compute_pixel_to_world_scale(minpt_bound, maxpt_bound, cameras);

  vector_3d diff = maxpt_bound - minpt_bound;
  vector_3d orig = minpt_bound;

  spacing = vector_3d(d_->grid_spacing);
  spacing *= pixel_to_world_scale * d_->voxel_spacing_factor;
  double max_spacing = spacing.maxCoeff();

  for (int i = 0; i < 3; i++)
  {
    d_->grid_dims[i] = static_cast<int>((diff[i] / spacing[i]));
  }

  LOG_DEBUG( logger(), "voxel size: " << spacing[0]
                       << " "         << spacing[1]
                       << " "         << spacing[2] );
  LOG_DEBUG( logger(), "grid: " << d_->grid_dims[0]
                       << " "   << d_->grid_dims[1]
                       << " "   << d_->grid_dims[2] );

  LOG_INFO( logger(), "initialize" );
  cuda_initalize(d_->grid_dims, orig.data(), spacing.data(),
                 d_->ray_potential_thickness * max_spacing,
                 d_->ray_potential_rho,
                 d_->ray_potential_eta,
                 d_->ray_potential_epsilon,
                 d_->ray_potential_delta * max_spacing);
  const size_t vsize = static_cast<size_t>(d_->grid_dims[0]) *
                       static_cast<size_t>(d_->grid_dims[1]) *
                       static_cast<size_t>(d_->grid_dims[2]);

  cuda_ptr<double> d_volume = init_volume_on_gpu(vsize);
  cuda_ptr<double> d_K = make_cuda_mem<double>(16);
  cuda_ptr<double> d_RT = make_cuda_mem<double>(16);

  for (size_t i = 0; i < depth_maps.size(); i++)
  {
    int depthmap_dims[2];
    depthmap_dims[0] = static_cast<int>(depth_maps[i]->width());
    depthmap_dims[1] = static_cast<int>(depth_maps[i]->height());
    cuda_ptr<double> d_depth = copy_img_to_gpu(depth_maps[i]);
    cuda_ptr<double> d_weight = nullptr;
    if (i < weight_maps.size())
    {
      auto weight = weight_maps[i];
      if (weight->width() == depth_maps[i]->width() &&
          weight->height() == depth_maps[i]->height())
      {
        d_weight = copy_img_to_gpu(weight_maps[i]);
      }
    }
    copy_camera_to_gpu(cameras[i], d_K.get(), d_RT.get());

    // run code on device
    LOG_INFO( logger(), "depth map " << i );
    launch_depth_kernel(d_depth.get(), d_weight.get(), depthmap_dims,
                        d_K.get(), d_RT.get(), d_volume.get(),
                        d_->max_voxels_per_launch);
  }

  // Transfer data from device to host
  auto h_volume = std::make_shared<image_memory>(vsize * sizeof(double));
  CudaErrorCheck(cudaMemcpy(h_volume->data(), d_volume.get(), vsize * sizeof(double),
                 cudaMemcpyDeviceToHost));

  volume = std::make_shared<simple_image_container>(
    image_of<double>(h_volume, reinterpret_cast<const double*>(h_volume->data()),
                    d_->grid_dims[0], d_->grid_dims[1], d_->grid_dims[2],
                    1, d_->grid_dims[0], d_->grid_dims[0] * d_->grid_dims[1]));
}

} // end namespace cuda
} // end namespace arrows
} // end namespace kwiver
