/*ckwg +29
* Copyright 2018, 2019 by Kitware, Inc.
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

/**
* \file
* \brief Implementation for kwiver::vital::sfm_constraints class storing
*        constraints to be used in SfM.
*/

#include <vital/types/sfm_constraints.h>

#include <vital/math_constants.h>
#include <vital/types/rotation.h>

namespace kwiver {
namespace vital {

/// Private implementation class
class sfm_constraints::priv
{
public:
  /// Constructor
  priv();

  /// Destructor
  ~priv();

  struct im_data {
    int width;
    int height;

    im_data() :
      width(-1),
      height(-1)
    {

    }
    im_data(int w_, int h_):
      width(w_),
      height(h_)
    {

    }
    im_data(const im_data& other):
      width(other.width),
      height(other.height)
    {

    }

  };

  metadata_map_sptr m_md;
  local_geo_cs m_lgcs;
  std::map<frame_id_t, im_data> m_image_data;
};

sfm_constraints::priv
::priv()
{

}

sfm_constraints::priv
::~priv()
{

}

sfm_constraints
::sfm_constraints(const sfm_constraints& other)
  : m_priv(new priv)
{
  m_priv->m_lgcs = other.m_priv->m_lgcs;
  m_priv->m_md = other.m_priv->m_md;
  m_priv->m_image_data = other.m_priv->m_image_data;
}

sfm_constraints
::sfm_constraints()
  : m_priv(new priv)
{

}

sfm_constraints
::sfm_constraints( metadata_map_sptr md,
                   local_geo_cs const& lgcs)
  :m_priv(new priv)
{
  m_priv->m_md = md;
  m_priv->m_lgcs = lgcs;
}

sfm_constraints
::~sfm_constraints()
{

}

metadata_map_sptr
sfm_constraints
::get_metadata()
{
  return m_priv->m_md;
}

void
sfm_constraints
::set_metadata(metadata_map_sptr md)
{
  m_priv->m_md = md;
}

local_geo_cs
sfm_constraints
::get_local_geo_cs()
{
  return m_priv->m_lgcs;
}

void
sfm_constraints
::set_local_geo_cs(local_geo_cs const& lgcs)
{
  m_priv->m_lgcs = lgcs;
}

bool
sfm_constraints
::get_focal_length_prior(frame_id_t fid, float &focal_length) const
{
  if (!m_priv->m_md)
  {
    return false;
  }

  auto &md = *m_priv->m_md;

  std::set<frame_id_t> frame_ids_to_try;

  int image_width = -1;
  if (!get_image_width(fid, image_width))
  {
    return false;
  }

  if (fid >= 0)
  {
    frame_ids_to_try.insert(fid);
  }
  else
  {
    frame_ids_to_try = md.frames();
  }

  std::vector<double> focal_lengths;
  for (auto test_fid : frame_ids_to_try)
  {
    if ( md.has<VITAL_META_SENSOR_HORIZONTAL_FOV>(test_fid) )
    {
      double hfov = md.get<VITAL_META_SENSOR_HORIZONTAL_FOV>(test_fid);
      focal_lengths.push_back(static_cast<float>(
        (image_width*0.5) / tan(0.5*hfov*deg_to_rad)));
      continue;
    }

    if ( md.has<VITAL_META_TARGET_WIDTH>(test_fid) &&
         md.has<VITAL_META_SLANT_RANGE>(test_fid) )
    {
      focal_length = static_cast<float>(image_width *
        md.get<VITAL_META_SLANT_RANGE>(test_fid) /
        md.get<VITAL_META_TARGET_WIDTH>(test_fid) );
      focal_lengths.push_back(focal_length);
      continue;
    }
  }
  if (focal_lengths.empty())
  {
    return false;
  }
  // compute the median focal length
  std::nth_element(focal_lengths.begin(),
                   focal_lengths.begin() + focal_lengths.size() / 2,
                   focal_lengths.end());
  focal_length = focal_lengths[focal_lengths.size() / 2];

  return true;
}

bool
sfm_constraints
::get_camera_orientation_prior_local(frame_id_t fid,
                                     rotation_d &R_loc) const
{
  if (m_priv->m_lgcs.origin().is_empty())
  {
    return false;
  }

  if (!m_priv->m_md)
  {
    return false;
  }

  auto &md = *m_priv->m_md;

  if ( md.has<VITAL_META_PLATFORM_HEADING_ANGLE>(fid) &&
       md.has<VITAL_META_PLATFORM_ROLL_ANGLE>(fid) &&
       md.has<VITAL_META_PLATFORM_PITCH_ANGLE>(fid) &&
       md.has<VITAL_META_SENSOR_REL_AZ_ANGLE>(fid) &&
       md.has<VITAL_META_SENSOR_REL_EL_ANGLE>(fid) )
  {
    double platform_heading = md.get<VITAL_META_PLATFORM_HEADING_ANGLE>(fid);
    double platform_roll = md.get<VITAL_META_PLATFORM_ROLL_ANGLE>(fid);
    double platform_pitch = md.get<VITAL_META_PLATFORM_PITCH_ANGLE>(fid);
    double sensor_rel_az = md.get<VITAL_META_SENSOR_REL_AZ_ANGLE>(fid);
    double sensor_rel_el = md.get<VITAL_META_SENSOR_REL_EL_ANGLE>(fid);

    double sensor_rel_roll = 0;
    if ( md.has<VITAL_META_SENSOR_REL_ROLL_ANGLE>(fid) )
    {
      sensor_rel_roll = md.get<VITAL_META_SENSOR_REL_ROLL_ANGLE>(fid);
    }

    if (std::isnan(platform_heading) || std::isnan(platform_pitch) || std::isnan(platform_roll) ||
        std::isnan(sensor_rel_az) || std::isnan(sensor_rel_el) || std::isnan(sensor_rel_roll))
    {
      return false;
    }

    R_loc = compose_rotations<double>(platform_heading, platform_pitch, platform_roll,
                                      sensor_rel_az, sensor_rel_el, sensor_rel_roll);

    return true;
  }

  return false;
}


bool
sfm_constraints
::get_camera_position_prior_local(frame_id_t fid,
                                  vector_3d &pos_loc) const
{
  if (m_priv->m_lgcs.origin().is_empty())
  {
    return false;
  }

  if (!m_priv->m_md)
  {
    return false;
  }

  kwiver::vital::geo_point gloc;
  if (m_priv->m_md->has<VITAL_META_SENSOR_LOCATION>(fid))
  {
    gloc = m_priv->m_md->get<VITAL_META_SENSOR_LOCATION>(fid);
  }
  else
  {
    return false;
  }

  auto geo_origin = m_priv->m_lgcs.origin();
  vector_3d loc = gloc.location(geo_origin.crs());
  loc -= geo_origin.location();

  pos_loc[0] = loc.x();
  pos_loc[1] = loc.y();
  pos_loc[2] = loc.z();

  return true;
}

sfm_constraints::position_map
sfm_constraints
::get_camera_position_priors() const
{
  position_map local_positions;

  if (!m_priv->m_md)
  {
    return local_positions;
  }

  for (auto mdv : m_priv->m_md->metadata())
  {
    auto fid = mdv.first;

    vector_3d loc;
    if (!get_camera_position_prior_local(fid, loc))
    {
      continue;
    }
    if (local_positions.empty())
    {
      local_positions[fid] = loc;
    }
    else
    {
      auto last_loc = local_positions.crbegin()->second;
      if (loc == last_loc)
      {
        continue;
      }
      local_positions[fid] = loc;
    }
  }
  return local_positions;
}

void
sfm_constraints
::store_image_size(frame_id_t fid, int image_width, int image_height)
{
  m_priv->m_image_data[fid] = priv::im_data(image_width, image_height);
}

bool
sfm_constraints
::get_image_height(frame_id_t fid, int &image_height) const
{
  if (fid >= 0)
  {
    auto data_it = m_priv->m_image_data.find(fid);
    if (data_it == m_priv->m_image_data.end())
    {
      return false;
    }
    image_height = data_it->second.height;
    return true;
  }
  else
  {
    if (m_priv->m_image_data.empty())
    {
      return false;
    }
    image_height = m_priv->m_image_data.begin()->second.height;
    return true;
  }
}

bool
sfm_constraints
::get_image_width(frame_id_t fid, int &image_width) const
{
  if (fid >= 0)
  {
    auto data_it = m_priv->m_image_data.find(fid);
    if (data_it == m_priv->m_image_data.end())
    {
      return false;
    }
    image_width = data_it->second.width;
    return true;
  }
  else
  {
    if (m_priv->m_image_data.empty())
    {
      return false;
    }
    image_width = m_priv->m_image_data.begin()->second.width;
    return true;
  }
}


}
}
