/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
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
 * \brief vxl estimate fundamental matrix implementation
 */

#include "estimate_fundamental_matrix.h"

#include <vital/util/enum_converter.h>
#include <vital/types/feature.h>

#include <arrows/vxl/camera.h>
#include <arrows/core/epipolar_geometry.h>

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
::check_configuration(vital::config_block_sptr config) const
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
  inliers = arrows::mark_fm_inliers(*fm, pts1, pts2, inlier_scale);
  return fm;
}


} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
