/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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
 * \brief Function to generate \ref kwiver::vital::camera_rpc from metadata
 */

#include "camera_from_metadata.h"

#include <vital/math_constants.h>
#include <vital/types/metadata_traits.h>

namespace kwiver {
namespace vital {

/// Extract scale or offset metadata to a vector
Eigen::VectorXd
tags_to_vector( metadata_sptr const& md, std::vector<vital_metadata_tag> tags )
{
  metadata_traits md_traits;
  auto vec_length = tags.size();

  Eigen::VectorXd rslt(vec_length);

  for (unsigned int i=0; i<vec_length; ++i)
  {
   if (auto& mdi = md->find(tags[i]))
    {
      rslt[i] = mdi.as_double();
    }
    else
    {
      throw metadata_exception("Missing RPC metadata: " +
                               md_traits.tag_to_name(tags[i]));
    }
  }

  return rslt;
}

/// Extract coefficient metadata to a matrix
rpc_matrix
tags_to_matrix( metadata_sptr const& md, std::vector<vital_metadata_tag> tags )
{
  metadata_traits md_traits;
  if (tags.size() != 4)
  {
    throw metadata_exception("Should have 4 metadata tags for RPC coefficients");
  }

  rpc_matrix rslt;

  for (int i=0; i<4; ++i)
  {
   if (auto& mdi = md->find(tags[i]))
    {
      rslt.row(i) = string_to_vector(mdi.as_string());
    }
    else
    {
      throw metadata_exception("Missing RPC metadata: " +
                               md_traits.tag_to_name(tags[i]));
    }
  }

  return rslt;
}

/// Convert space separated strings to Eigen vector
Eigen::VectorXd
VITAL_EXPORT string_to_vector( std::string const& s )
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, ' '))
  {
     tokens.push_back(token);
  }

  Eigen::VectorXd result(tokens.size());
  for (unsigned int i = 0; i<tokens.size(); ++i)
  {
    result[i] = std::stod(tokens[i]);
  }

  return result;
}

/// Produce RPC camera from metadata
camera_sptr
VITAL_EXPORT camera_from_metadata( metadata_sptr const& md )
{
  vector_3d world_scale, world_offset;
  vector_2d image_scale, image_offset;
  rpc_matrix rpc_coeffs;

  metadata_traits md_traits;

  std::vector<vital_metadata_tag> world_scale_tags = {
    vital::VITAL_META_RPC_LONG_SCALE,
    vital::VITAL_META_RPC_LAT_SCALE,
    vital::VITAL_META_RPC_HEIGHT_SCALE
  };
  world_scale = tags_to_vector(md, world_scale_tags);

  std::vector<vital_metadata_tag> world_offset_tags = {
    vital::VITAL_META_RPC_LONG_OFFSET,
    vital::VITAL_META_RPC_LAT_OFFSET,
    vital::VITAL_META_RPC_HEIGHT_OFFSET
  };
  world_offset = tags_to_vector(md, world_offset_tags);

  std::vector<vital_metadata_tag> image_scale_tags = {
    vital::VITAL_META_RPC_ROW_SCALE,
    vital::VITAL_META_RPC_COL_SCALE
  };
  image_scale = tags_to_vector(md, image_scale_tags);

  std::vector<vital_metadata_tag> image_offset_tags = {
    vital::VITAL_META_RPC_ROW_OFFSET,
    vital::VITAL_META_RPC_COL_OFFSET
  };
  image_offset = tags_to_vector(md, image_offset_tags);

  std::vector<vital_metadata_tag> rpc_coeffs_tags = {
    vital::VITAL_META_RPC_ROW_NUM_COEFF,
    vital::VITAL_META_RPC_ROW_DEN_COEFF,
    vital::VITAL_META_RPC_COL_NUM_COEFF,
    vital::VITAL_META_RPC_COL_DEN_COEFF
  };
  rpc_coeffs = tags_to_matrix(md, rpc_coeffs_tags);

  return std::make_shared<simple_camera_rpc>(world_scale, world_offset,
                                             image_scale, image_offset,
                                             rpc_coeffs);
}

/// Use metadata to construct intrinsics
VITAL_EXPORT
camera_intrinsics_sptr
intrinsics_from_metadata(metadata const& md,
                         unsigned int image_width,
                         unsigned int image_height)
{
  double im_w = static_cast<double>(image_width);
  double im_h = static_cast<double>(image_height);
  double focal_len = 0;

  auto& md_slant_range =
    md.find( vital::VITAL_META_SLANT_RANGE );
  auto& md_target_width =
    md.find( vital::VITAL_META_TARGET_WIDTH );
  if ( md_slant_range && md_target_width )
  {
    focal_len =
      im_w * (md_slant_range.as_double() / md_target_width.as_double());
  }
  else
  {
    auto& md_hfov =
      md.find( vital::VITAL_META_SENSOR_HORIZONTAL_FOV );
    if ( md_hfov )
    {
      focal_len =
        ( im_w / 2.0 ) / tan( 0.5 * md_hfov.as_double() * deg_to_rad );
    }
    else
    {
      return nullptr;
    }
  }

  vital::vector_2d pp(0.5*im_w, 0.5*im_h);
  return std::make_shared<simple_camera_intrinsics>
    (focal_len, pp, 1.0, 0.0, Eigen::VectorXd(), image_width, image_height);
}


/// Use a sequence of metadata objects to initialize a sequence of cameras
std::map<frame_id_t, camera_sptr>
initialize_cameras_with_metadata(std::map<frame_id_t, metadata_sptr> const& md_map,
                                 simple_camera_perspective const& base_camera,
                                 local_geo_cs& lgcs,
                                 rotation_d const& rot_offset)
{
  std::map<frame_id_t, camera_sptr> cam_map;
  vital::vector_3d mean(0, 0, 0);
  simple_camera_perspective active_cam(base_camera);

  bool update_local_origin = false;
  if (lgcs.origin().is_empty() && !md_map.empty())
  {
    // if a local coordinate system has not been established,
    // use the coordinates of the first camera
    for (auto m : md_map)
    {
      if (!m.second)
      {
        continue;
      }
      if(auto& mdi = m.second->find(vital::VITAL_META_SENSOR_LOCATION))
      {
        vital::geo_point gloc;
        mdi.data(gloc);

        lgcs.set_origin(gloc);
        lgcs.set_origin_altitude(0.0);
        update_local_origin = true;
        break;
      }
    }
  }
  for (auto const& p : md_map)
  {
    auto md = p.second;
    if (!md)
    {
      continue;
    }
    auto K = base_camera.get_intrinsics();
    K = intrinsics_from_metadata(*md, K->image_width(), K->image_height());
    if (K)
    {
      active_cam.set_intrinsics(K);
    }
    if (lgcs.update_camera(*md, active_cam, rot_offset))
    {
      mean += active_cam.center();
      cam_map[p.first] = camera_sptr(new simple_camera_perspective(active_cam));
    }
  }

  if (update_local_origin)
  {
    mean /= static_cast<double>(cam_map.size());
    // only use the mean easting and northing
    mean[2] = 0.0;

    // shift the UTM origin to the mean of the cameras easting and northing
    vector_2d mean_xy(mean.x(), mean.y());
    lgcs.set_origin(geo_point(lgcs.origin().location() + mean_xy, lgcs.origin().crs()));

    // shift all cameras to the new coordinate system.
    typedef std::map<frame_id_t, camera_sptr>::value_type cam_map_val_t;
    for (cam_map_val_t const &p : cam_map)
    {
      simple_camera_perspective* cam = dynamic_cast<simple_camera_perspective*>(p.second.get());
      cam->set_center(cam->get_center() - mean);
    }
  }

  return cam_map;
}


/// Update a sequence of metadata from a sequence of cameras and local_geo_cs
void
update_metadata_from_cameras(std::map<frame_id_t, camera_sptr> const& cam_map,
                             local_geo_cs const& lgcs,
                             std::map<frame_id_t, metadata_sptr>& md_map)
{
  if (lgcs.origin().is_empty())
  {
    // TODO throw an exception here?
    vital::logger_handle_t
      logger(vital::get_logger("update_metadata_from_cameras"));
    LOG_WARN(logger, "local geo coordinates do not have an origin");
    return;
  }

  typedef std::map<frame_id_t, camera_sptr>::value_type cam_map_val_t;
  for (cam_map_val_t const &p : cam_map)
  {
    auto active_md = md_map[p.first];
    if (!active_md)
    {
      md_map[p.first] = active_md = std::make_shared<vital::metadata>();
    }
    auto cam = dynamic_cast<vital::simple_camera_perspective*>(p.second.get());
    if (active_md && cam)
    {
      lgcs.update_metadata(*cam, *active_md);
    }
  }
}


} } // end of namespace
