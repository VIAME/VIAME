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
tags_to_vector( metadata_sptr const& md,
                std::vector<vital_metadata_tag> tags )
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
      VITAL_THROW(metadata_exception, "Missing RPC metadata: " +
                                      md_traits.tag_to_name(tags[i]));
    }
  }

  return rslt;
}

/// Extract coefficient metadata to a matrix
rpc_matrix
tags_to_matrix( metadata_sptr const& md,
                std::vector<vital_metadata_tag> tags )
{
  metadata_traits md_traits;
  if (tags.size() != 4)
  {
    VITAL_THROW(metadata_exception,
                "Should have 4 metadata tags for RPC coefficients");
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
      VITAL_THROW(metadata_exception, "Missing RPC metadata: " +
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
    VITAL_META_RPC_LONG_SCALE,
    VITAL_META_RPC_LAT_SCALE,
    VITAL_META_RPC_HEIGHT_SCALE
  };
  world_scale = tags_to_vector(md, world_scale_tags);

  std::vector<vital_metadata_tag> world_offset_tags = {
    VITAL_META_RPC_LONG_OFFSET,
    VITAL_META_RPC_LAT_OFFSET,
    VITAL_META_RPC_HEIGHT_OFFSET
  };
  world_offset = tags_to_vector(md, world_offset_tags);

  std::vector<vital_metadata_tag> image_scale_tags = {
    VITAL_META_RPC_ROW_SCALE,
    VITAL_META_RPC_COL_SCALE
  };
  image_scale = tags_to_vector(md, image_scale_tags);

  std::vector<vital_metadata_tag> image_offset_tags = {
    VITAL_META_RPC_ROW_OFFSET,
    VITAL_META_RPC_COL_OFFSET
  };
  image_offset = tags_to_vector(md, image_offset_tags);

  std::vector<vital_metadata_tag> rpc_coeffs_tags = {
    VITAL_META_RPC_ROW_NUM_COEFF,
    VITAL_META_RPC_ROW_DEN_COEFF,
    VITAL_META_RPC_COL_NUM_COEFF,
    VITAL_META_RPC_COL_DEN_COEFF
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
    md.find( VITAL_META_SLANT_RANGE );
  auto& md_target_width =
    md.find( VITAL_META_TARGET_WIDTH );
  if ( md_slant_range && md_target_width )
  {
    focal_len =
      im_w * (md_slant_range.as_double() / md_target_width.as_double());
  }
  else
  {
    auto& md_hfov =
      md.find( VITAL_META_SENSOR_HORIZONTAL_FOV );
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

  vector_2d pp(0.5*im_w, 0.5*im_h);
  return std::make_shared<simple_camera_intrinsics>
    (focal_len, pp, 1.0, 0.0, Eigen::VectorXd(), image_width, image_height);
}


/// Use a sequence of metadata objects to initialize a sequence of cameras
std::map<frame_id_t, camera_sptr>
initialize_cameras_with_metadata(std::map<frame_id_t,
                                          metadata_sptr> const& md_map,
                                 simple_camera_perspective const& base_camera,
                                 local_geo_cs& lgcs,
                                 bool init_intrinsics,
                                 rotation_d const& rot_offset)
{
  std::map<frame_id_t, camera_sptr> cam_map;
  vector_3d mean(0, 0, 0);
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
      if(auto& mdi = m.second->find(VITAL_META_SENSOR_LOCATION))
      {
        geo_point gloc;
        mdi.data(gloc);

        // set the origin to the ground
        vital::vector_3d loc = gloc.location();
        loc[2] = 0.0;
        gloc.set_location(loc, gloc.crs());

        lgcs.set_origin(gloc);
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
    if (init_intrinsics)
    {
      auto K = base_camera.get_intrinsics();
      K = intrinsics_from_metadata(*md, K->image_width(), K->image_height());
      if (K)
      {
        active_cam.set_intrinsics(K);
      }
    }
    if (update_camera_from_metadata(*md, lgcs, active_cam, rot_offset))
    {
      mean += active_cam.center();
      cam_map[p.first] =
        std::make_shared<simple_camera_perspective>(active_cam);
    }
  }

  if (update_local_origin)
  {
    mean /= static_cast<double>(cam_map.size());
    // only use the mean easting and northing
    mean[2] = 0.0;

    // shift the UTM origin to the mean of the cameras easting and northing
    vital::vector_3d offset = lgcs.origin().location() + mean;
    lgcs.set_origin(geo_point(offset, lgcs.origin().crs()));

    // shift all cameras to the new coordinate system.
    typedef std::map<frame_id_t, camera_sptr>::value_type cam_map_val_t;
    for (cam_map_val_t const &p : cam_map)
    {
      simple_camera_perspective* cam =
        dynamic_cast<simple_camera_perspective*>(p.second.get());
      cam->set_center(cam->get_center() - mean);
    }
  }

  return cam_map;
}


/// Use the pose data provided by metadata to update camera pose
bool
update_camera_from_metadata(metadata const& md,
                            local_geo_cs const& lgcs,
                            simple_camera_perspective& cam,
                            rotation_d const& rot_offset)
{
  bool rotation_set = false;
  bool translation_set = false;

  bool has_platform_yaw = false;
  bool has_platform_pitch = false;
  bool has_platform_roll = false;
  bool has_sensor_yaw = false;
  bool has_sensor_pitch = false;

  double platform_yaw = 0.0, platform_pitch = 0.0, platform_roll = 0.0;
  if (auto& mdi = md.find(VITAL_META_PLATFORM_HEADING_ANGLE))
  {
    mdi.data(platform_yaw);
    has_platform_yaw = true;
  }
  if (auto& mdi = md.find(VITAL_META_PLATFORM_PITCH_ANGLE))
  {
    mdi.data(platform_pitch);
    has_platform_pitch = true;
  }
  if (auto& mdi = md.find(VITAL_META_PLATFORM_ROLL_ANGLE))
  {
    mdi.data(platform_roll);
    has_platform_roll = true;
  }
  double sensor_yaw = 0.0, sensor_pitch = 0.0, sensor_roll = 0.0;
  if (auto& mdi = md.find(VITAL_META_SENSOR_REL_AZ_ANGLE))
  {
    mdi.data(sensor_yaw);
    has_sensor_yaw = true;
  }
  if (auto& mdi = md.find(VITAL_META_SENSOR_REL_EL_ANGLE))
  {
    mdi.data(sensor_pitch);
    has_sensor_pitch = true;
  }
  if (auto& mdi = md.find(VITAL_META_SENSOR_REL_ROLL_ANGLE))
  {
    mdi.data(sensor_roll);
  }


  if (has_platform_yaw && has_platform_pitch && has_platform_roll &&
      has_sensor_yaw && has_sensor_pitch &&
      // Sensor roll is ignored here on purpose.
      // It is fixed on some platforms to zero.
      !(std::isnan(platform_yaw) || std::isnan(platform_pitch) ||
        std::isnan(platform_roll) || std::isnan(sensor_yaw) ||
        std::isnan(sensor_pitch) || std::isnan(sensor_roll)))
  {
    //only set the camera's rotation if all metadata angles are present

    auto R = compose_rotations<double>(
      platform_yaw, platform_pitch, platform_roll,
      sensor_yaw, sensor_pitch, sensor_roll);

    cam.set_rotation(R);

    rotation_set = true;
  }

  if (auto& mdi = md.find(VITAL_META_SENSOR_LOCATION))
  {
    geo_point gloc;
    mdi.data(gloc);

    // get the location in the same UTM zone as the origin
    vector_3d loc = gloc.location(lgcs.origin().crs())
                  - lgcs.origin().location();
    cam.set_center(loc);
    translation_set = true;
  }
  return rotation_set || translation_set;
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
    logger_handle_t
      logger(get_logger("update_metadata_from_cameras"));
    LOG_WARN(logger, "local geo coordinates do not have an origin");
    return;
  }

  typedef std::map<frame_id_t, camera_sptr>::value_type cam_map_val_t;
  for (cam_map_val_t const &p : cam_map)
  {
    auto active_md = md_map[p.first];
    if (!active_md)
    {
      md_map[p.first] = active_md = std::make_shared<metadata>();
    }
    auto cam = dynamic_cast<simple_camera_perspective*>(p.second.get());
    if (active_md && cam)
    {
      update_metadata_from_camera(*cam, lgcs, *active_md);
    }
  }
}


/// Use the camera pose to update the metadata structure
void
update_metadata_from_camera(simple_camera_perspective const& cam,
                            local_geo_cs const& lgcs,
                            metadata& md)
{
  if (md.has(VITAL_META_PLATFORM_HEADING_ANGLE) &&
      md.has(VITAL_META_PLATFORM_PITCH_ANGLE) &&
      md.has(VITAL_META_PLATFORM_ROLL_ANGLE) &&
      md.has(VITAL_META_SENSOR_REL_AZ_ANGLE) &&
      md.has(VITAL_META_SENSOR_REL_EL_ANGLE))
  {
    // We have a complete metadata rotation.
    // Note that sensor roll is ignored here on purpose.
    double yaw, pitch, roll;
    cam.rotation().get_yaw_pitch_roll(yaw, pitch, roll);
    yaw *= rad_to_deg;
    pitch *= rad_to_deg;
    roll *= rad_to_deg;
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_YAW_ANGLE, yaw));
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_PITCH_ANGLE, pitch));
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_ROLL_ANGLE, roll));
  }

  if (md.has(VITAL_META_SENSOR_LOCATION))
  {
    // we have a complete position from metadata.
    const vector_3d loc = cam.get_center() + lgcs.origin().location();
    geo_point gc(loc, lgcs.origin().crs());

    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_LOCATION, gc));
  }
}


} } // end of namespace
