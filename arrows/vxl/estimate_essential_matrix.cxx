// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vxl estimate essential matrix implementation
 */

#include "estimate_essential_matrix.h"

#include <vital/types/feature.h>
#include <vital/vital_config.h>

#include <arrows/vxl/camera.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <vgl/vgl_point_2d.h>

#include <vpgl/algo/vpgl_em_compute_5_point.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

/// Private implementation class
class estimate_essential_matrix::priv
{
public:
  /// Constructor
  priv()
  : verbose(false),
    num_ransac_samples(512)
  {
  }

  bool verbose;
  unsigned num_ransac_samples;
};

/// Constructor
estimate_essential_matrix
::estimate_essential_matrix()
: d_(new priv)
{
}

/// Destructor
estimate_essential_matrix
::~estimate_essential_matrix()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
estimate_essential_matrix
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::estimate_essential_matrix::get_configuration();

  config->set_value("verbose", d_->verbose,
                    "If true, write status messages to the terminal showing "
                    "debugging information");

  config->set_value("num_ransac_samples", d_->num_ransac_samples,
                    "The number of samples to use in RANSAC");

  return config;
}

/// Set this algorithm's properties via a config block
void
estimate_essential_matrix
::set_configuration(vital::config_block_sptr config)
{

  d_->verbose = config->get_value<bool>("verbose",
                                        d_->verbose);
  d_->num_ransac_samples = config->get_value<unsigned>("num_ransac_samples",
                                                       d_->num_ransac_samples);
}

/// Check that the algorithm's currently configuration is valid
bool
estimate_essential_matrix
::check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
}

/// Estimate an essential matrix from corresponding points
essential_matrix_sptr
estimate_essential_matrix
::estimate(const std::vector<vector_2d>& pts1,
           const std::vector<vector_2d>& pts2,
           const camera_intrinsics_sptr cal1,
           const camera_intrinsics_sptr cal2,
           std::vector<bool>& inliers,
           double inlier_scale) const
{
  vpgl_calibration_matrix<double> vcal1, vcal2;
  vital_to_vpgl_calibration(*cal1, vcal1);
  vital_to_vpgl_calibration(*cal2, vcal2);

  std::vector<vgl_point_2d<double> > right_points, left_points;
  for(const vector_2d& v : pts1)
  {
    right_points.push_back(vgl_point_2d<double>(v.x(), v.y()));
  }
  for(const vector_2d& v : pts2)
  {
    left_points.push_back(vgl_point_2d<double>(v.x(), v.y()));
  }

  double sq_scale = inlier_scale * inlier_scale;
  vpgl_em_compute_5_point_ransac<double> em(d_->num_ransac_samples, sq_scale,
                                            d_->verbose);
  vpgl_essential_matrix<double> best_em;
  em.compute(right_points, vcal1, left_points, vcal2, best_em);

  matrix_3x3d E(best_em.get_matrix().data_block());
  E.transposeInPlace();
  matrix_3x3d K1_inv = cal1->as_matrix().inverse();
  matrix_3x3d K2_invt = cal2->as_matrix().transpose().inverse();
  matrix_3x3d F = K2_invt * E * K1_inv;

  fundamental_matrix_sptr fm(new fundamental_matrix_d(F));
  inliers = mvg::mark_fm_inliers(*fm, pts1, pts2, inlier_scale);

  return essential_matrix_sptr(new essential_matrix_d(E));
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
