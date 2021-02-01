// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vxl estimate fundamental matrix implementation
 */

#include "estimate_fundamental_matrix.h"

#include <vital/util/enum_converter.h>
#include <vital/types/feature.h>
#include <vital/vital_config.h>

#include <arrows/vxl/camera.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <vgl/vgl_point_2d.h>
#include <Eigen/LU>

#include <vpgl/algo/vpgl_fm_compute_8_point.h>
#include <vpgl/algo/vpgl_fm_compute_7_point.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

namespace {
  enum method_t { EST_7_POINT, EST_8_POINT };
}

/// Private implementation class
class estimate_fundamental_matrix::priv
{
public:
  /// Constructor
  priv()
  : precondition(true),
    method(EST_8_POINT)
  {
  }

  bool precondition;
  method_t method;
};

// Define the enum converter
ENUM_CONVERTER( method_converter, method_t,
                { "EST_7_POINT",   EST_7_POINT },
                { "EST_8_POINT",   EST_8_POINT }
)

/// Constructor
estimate_fundamental_matrix
::estimate_fundamental_matrix()
: d_(new priv)
{
}

/// Destructor
estimate_fundamental_matrix
::~estimate_fundamental_matrix()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
estimate_fundamental_matrix
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::estimate_fundamental_matrix::get_configuration();

  config->set_value("precondition", d_->precondition,
                    "If true, precondition the data before estimating the "
                    "fundamental matrix");

  config->set_value("method", method_converter().to_string( d_->method ),
                    "Fundamental matrix estimation method to use. "
                    "(Note: does not include RANSAC).  Choices are: "
                    + method_converter().element_name_string() );

  return config;
}

/// Set this algorithm's properties via a config block
void
estimate_fundamental_matrix
::set_configuration(vital::config_block_sptr config)
{

  d_->precondition = config->get_value<bool>("precondition",
                                             d_->precondition);

  d_->method = config->get_enum_value< method_converter >( "method",
                                                           d_->method );
}

/// Check that the algorithm's current configuration is valid
bool
estimate_fundamental_matrix
::check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
}

/// Estimate an essential matrix from corresponding points
fundamental_matrix_sptr
estimate_fundamental_matrix
::estimate(const std::vector<vector_2d>& pts1,
           const std::vector<vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale) const
{
  std::vector<vgl_homg_point_2d<double> > right_points, left_points;
  for(const vector_2d& v : pts1)
  {
    right_points.push_back(vgl_homg_point_2d<double>(v.x(), v.y()));
  }
  for(const vector_2d& v : pts2)
  {
    left_points.push_back(vgl_homg_point_2d<double>(v.x(), v.y()));
  }

  vpgl_fundamental_matrix<double> vfm;
  if( d_->method == EST_8_POINT )
  {
    vpgl_fm_compute_8_point fm_compute(d_->precondition);
    fm_compute.compute(right_points, left_points, vfm);
  }
  else
  {
    std::vector< vpgl_fundamental_matrix<double>* > vfms;
    vpgl_fm_compute_7_point fm_compute(d_->precondition);
    fm_compute.compute(right_points, left_points, vfms);
    // TODO use the multiple solutions in a RANSAC framework
    // For now, only keep the first solution
    vfm = *vfms[0];
    for(auto v : vfms)
    {
      delete v;
    }
  }

  matrix_3x3d F(vfm.get_matrix().data_block());
  F.transposeInPlace();

  fundamental_matrix_sptr fm(new fundamental_matrix_d(F));
  inliers = mvg::mark_fm_inliers(*fm, pts1, pts2, inlier_scale);
  return fm;
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
