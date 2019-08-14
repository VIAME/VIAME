/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "transform_detected_object_set.h"

#include <iostream>
#include <vital/io/camera_io.h>
#include <vital/config/config_difference.h>
#include <vital/util/string.h>
#include <vital/types/bounding_box.h>
#include <Eigen/Core>

namespace kwiver {
namespace arrows {
namespace core {

// ------------------------------------------------------------------
transform_detected_object_set::transform_detected_object_set()
  : src_camera_krtd_file_name( "" )
  , dest_camera_krtd_file_name( "" )
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
transform_detected_object_set::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "src_camera_krtd_file_name", src_camera_krtd_file_name,
                     "Source camera KRTD file name path" );

  config->set_value( "dest_camera_krtd_file_name", dest_camera_krtd_file_name,
                     "Destination camera KRTD file name path" );

  return config;
}


// ------------------------------------------------------------------
void
transform_detected_object_set::
set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );
  this->src_camera_krtd_file_name = config->get_value< std::string > ( "src_camera_krtd_file_name" );
  this->dest_camera_krtd_file_name = config->get_value< std::string > ( "dest_camera_krtd_file_name" );

  this->src_camera = kwiver::vital::read_krtd_file( this->src_camera_krtd_file_name );
  this->dest_camera = kwiver::vital::read_krtd_file( this->dest_camera_krtd_file_name );
}


// ------------------------------------------------------------------
bool
transform_detected_object_set::
check_configuration( vital::config_block_sptr config ) const
{
  kwiver::vital::config_difference cd( this->get_configuration(), config );
  const auto key_list = cd.extra_keys();

  if ( ! key_list.empty() )
  {
    LOG_WARN( logger(), "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
  }

  return true;
}


// ------------------------------------------------------------------
Eigen::Vector3d
transform_detected_object_set::
backproject_to_height( const kwiver::vital::camera_perspective_sptr camera,
		       const Eigen::Vector2d img_pt ) const
{
  // Back an image point to a specified world height
  double height = 0.0;

  Eigen::Vector3d img_pt_3;
  img_pt_3 << img_pt, 1.0;

  Eigen::Vector3d npt = camera->intrinsics()->as_matrix().colPivHouseholderQr().solve( img_pt_3 );
  Eigen::Matrix3d M = camera->rotation().matrix();
  Eigen::Vector3d t = camera->translation();

  M(0,2) = height * M(0,2) + t(0);
  M(1,2) = height * M(1,2) + t(1);
  M(2,2) = height * M(2,2) + t(2);

  Eigen::Vector3d wpt = M.colPivHouseholderQr().solve( npt );
  auto output = Eigen::Vector3d(wpt(0) / wpt(2), wpt(1) / wpt(2), height);

  return output;
}

// ------------------------------------------------------------------
Eigen::Vector3d
transform_detected_object_set::
backproject_to_plane( const kwiver::vital::camera_perspective_sptr camera,
		      const Eigen::Vector2d img_pt,
		      const Eigen::Vector4d plane ) const
{
  // Back an image point to a specified world plane
  Eigen::Vector3d img_pt_3;
  img_pt_3 << img_pt, 1.0;

  Eigen::Vector3d npt = camera->intrinsics()->as_matrix().colPivHouseholderQr().solve( img_pt_3 );
  Eigen::Matrix3d M = camera->rotation().matrix().transpose();

  auto n = Eigen::Vector3d(plane(0), plane(1), plane(2));
  double d = plane(3);

  Eigen::Vector3d Mt = M * camera->translation();
  Eigen::Vector3d Mp = M * npt;

  return Mp * (n.dot( Mt ) - d) / n.dot( Mp ) - Mt;
}

// ------------------------------------------------------------------
Eigen::Matrix<double, 8, 3>
transform_detected_object_set::
backproject_bbox( const kwiver::vital::camera_perspective_sptr camera,
		  const Eigen::Vector4d box ) const
{
  // project base of the box to the ground
  auto base_pt = Eigen::Vector2d((box(0) + box(2)) / 2, box(3));
  Eigen::Vector3d pc = this->backproject_to_height( camera, base_pt );
  Eigen::Vector3d ray = pc - (-camera->rotation().matrix().transpose() * camera->translation());
  ray(2) = 0.0;
  ray.normalize();

  Eigen::Vector3d p1 = this->backproject_to_height( camera, Eigen::Vector2d(box(0), box(3)) );
  Eigen::Vector3d p2 = this->backproject_to_height( camera, Eigen::Vector2d(box(2), box(3)) );

  Eigen::Vector3d vh = p2 - p1;
  Eigen::Vector3d vd = ray * vh.norm();
  vh = Eigen::Vector3d(-vd(1), vd(0), 0);
  p1 = pc - vh / 2;
  p2 = pc + vh / 2;
  Eigen::Vector3d p3 = p2 + vd;
  Eigen::Vector3d p4 = p1 + vd;

  // TODO: Check that norm of vd is not equal to 0.0?

  Eigen::Vector3d n = vd.normalized();
  double d = -(n.transpose() * p3)(0);

  Eigen::Vector4d back_plane;
  back_plane << n, d;
  Eigen::Vector3d p5 = this->backproject_to_plane( camera, Eigen::Vector2d(box(0), box(1)), back_plane );
  double height = p5(2);

  Eigen::Matrix<double, 8, 3> box3d;
  box3d << p1.transpose(),
    p2.transpose(),
    p3.transpose(),
    p4.transpose(),
    p1(0), p1(1), height,
    p2(0), p2(1), height,
    p3(0), p3(1), height,
    p4(0), p4(1), height;

  return box3d;
}

// ------------------------------------------------------------------
Eigen::Vector4d
transform_detected_object_set::
box_around_box3d( const kwiver::vital::camera_perspective_sptr camera,
		  const Eigen::Matrix<double, 8, 3> box3d ) const
{
  Eigen::Matrix<double, 8, 2> projected_points;

  // project the points
  for (int i=0; i<8; i++) {
    projected_points.row(i) = camera->project(box3d.row(i)).transpose();
  }

  Eigen::Vector4d out_box;
  out_box << (Eigen::Vector2d)projected_points.colwise().minCoeff(),
    (Eigen::Vector2d)projected_points.colwise().maxCoeff();

  return out_box;
}

// ------------------------------------------------------------------
Eigen::Vector4d
transform_detected_object_set::
view_to_view( const kwiver::vital::camera_perspective_sptr src_camera,
	      const kwiver::vital::camera_perspective_sptr dest_camera,
	      const Eigen::Vector4d bounds ) const
{
  Eigen::Matrix<double, 8, 3> box3d = this->backproject_bbox( src_camera, bounds );
  Eigen::Vector4d tgt_box = this->box_around_box3d( dest_camera, box3d );

  return tgt_box;
}

// ------------------------------------------------------------------
vital::bounding_box<double>
transform_detected_object_set::
transform_bounding_box( vital::bounding_box<double>& bbox ) const
{
  Eigen::Vector4d bbox_bounds;
  bbox_bounds << bbox.min_x(), bbox.min_y(),
    bbox.max_x(), bbox.max_y();

  Eigen::Vector4d out_bounds = this->view_to_view( this->src_camera,
						   this->dest_camera,
						   bbox_bounds );

  return vital::bounding_box<double>(out_bounds(0),
				     out_bounds(1),
				     out_bounds(2),
				     out_bounds(3));
}

// ------------------------------------------------------------------
vital::detected_object_set_sptr
transform_detected_object_set::
filter( const vital::detected_object_set_sptr input_set ) const
{
  auto ret_set = std::make_shared<vital::detected_object_set>();

  // loop over all detections
  auto ie = input_set->cend();
  for ( auto det = input_set->cbegin(); det != ie; ++det )
  {
    auto out_det = (*det)->clone();
    auto out_box = out_det->bounding_box();
    auto new_out_box = this->transform_bounding_box(out_box);
    out_det->set_bounding_box( new_out_box );
    ret_set->add( out_det );
  } // end foreach detection

  return ret_set;
} // transform_detected_object_set::filter

} } }     // end namespace
