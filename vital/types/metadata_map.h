/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Header file for a map from frame IDs to metadata vectors
 */

#ifndef KWIVER_VITAL_METADATA_MAP_H_
#define KWIVER_VITAL_METADATA_MAP_H_

#include <vital/types/metadata.h>

#include <vital/vital_types.h>
#include <vital/vital_config.h>
#include <vital/types/geo_point.h>

#include <map>
#include <memory>
#include <set>

namespace kwiver {
namespace vital {

/// An abstract mapping between frame IDs and metadata vectors
/*
 * \note a vector of metadata objects is used because each frame could
 * have multiple metadata blocks.  For example, metadata may come from
 * multiple sources on a given frame or a metadata may be provided at
 * a higher sampling rate than the video sampling rate.
 */
class metadata_map
{
public:
  /// typedef for std::map from integer frame IDs to metadata vectors
  typedef std::map< frame_id_t, metadata_vector > map_metadata_t;

  /// Destructor
  virtual ~metadata_map() = default;

  /// Return the number of frames in the map
  virtual size_t size() const = 0;

  /// Return a map from integer frame IDs to metadata vectors
  virtual map_metadata_t metadata() const = 0;

  /// gets the location of the sensor from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] loc the geo location
  * \returns true if location was found in metadata and set, false otherwise
  */
  virtual bool get_sensor_location(frame_id_t fid, geo_point &loc) = 0;

  /// gets the altitiude of the sensor from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] altitude the sensor's altitude
  * \returns true if altitude was found in metadata and set, false otherwise
  */
  virtual bool get_sensor_altitude(frame_id_t fid, double &altitude) = 0;

  /// gets the horizontal field of view of the sensor from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] fov the field of view 
  * \returns true if field of view was found in metadata and set, false otherwise
  */
  virtual bool get_horizontal_field_of_view(frame_id_t fid, double &fov) = 0;

  /// gets the slant range to the target from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] slant range the range to the target
  * \returns true if slant range was found in metadata and set, false otherwise
  */
  virtual bool get_slant_range(frame_id_t fid, double &slant_range) = 0;

  /// gets the target width from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] target_width the width of the target
  * \returns true if target width was found in metadata and set, false otherwise
  */
  virtual bool get_target_width(frame_id_t fid, double &target_width) = 0;

  /// gets the platform's heading from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] heading the platform's heading
  * \returns true if the platform's heading was found in metadata and set, false otherwise
  */
  virtual bool get_platform_heading_angle(frame_id_t fid, double &heading) = 0;

  /// gets the platform's pitch from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] pitch the platform's pitch
  * \returns true if the platform's pitch was found in metadata and set, false otherwise
  */
  virtual bool get_platform_pitch_angle(frame_id_t fid, double &pitch) = 0;

  /// gets the platform's roll from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] roll the platform's roll
  * \returns true if the platform's roll was found in metadata and set, false otherwise
  */
  virtual bool get_platform_roll_angle(frame_id_t fid, double &roll) = 0;

  /// gets the sensor's azimuth angle from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] rel_az_angle the sensor's azimuth angle relative to the platform
  * \returns true if the azimuth angle was found in metadata and set, false otherwise
  */
  virtual bool get_sensor_rel_az_angle(frame_id_t fid, double &rel_az_angle) = 0;

  /// gets the sensor's elevation angle from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] rel_el_angle the sensor's elevation angle relative to the platform
  * \returns true if the elevation angle was found in metadata and set, false otherwise
  */
  virtual bool get_sensor_rel_el_angle(frame_id_t fid, double &rel_el_angle) = 0;

  /// gets the sensor's roll angle from the metadata
  /**
  * \param[in] fid the frame id
  * \param[out] rel_roll_angle the sensor's elevation roll relative to the platform
  * \returns true if the roll angle was found in metadata and set, false otherwise
  */
  virtual bool get_sensor_rel_roll_angle(frame_id_t fid, double &rel_roll_angle) = 0;

  virtual std::set<frame_id_t> frames() = 0;

};

/// typedef for a metadata shared pointer
typedef std::shared_ptr< metadata_map > metadata_map_sptr;


/// A concrete metadata_map that simply wraps a std::map.
class simple_metadata_map :
  public metadata_map
{
public:
  /// Default Constructor
  simple_metadata_map() { }

  /// Constructor from a std::map of metadata
  explicit simple_metadata_map( map_metadata_t const& metadata )
    : data_( metadata ) { }

  /// Return the number of metadata in the map
  virtual size_t size() const { return data_.size(); }

  /// Return a map from integer IDs to metadata shared pointers
  virtual map_metadata_t metadata() const { return data_; }

  virtual bool get_sensor_location(frame_id_t fid, geo_point &loc)
  {
    return get_value<geo_point>(VITAL_META_SENSOR_LOCATION,fid, loc);
  }
  
  virtual bool get_sensor_altitude(frame_id_t fid, double &altitude)
  {
    return get_value<double>(VITAL_META_SENSOR_ALTITUDE, fid, altitude);
  }

  virtual bool get_horizontal_field_of_view(frame_id_t fid, double &fov)
  {
    return get_value<double>(VITAL_META_SENSOR_HORIZONTAL_FOV, fid, fov);
  }

  virtual bool get_slant_range(frame_id_t fid, double &slant_range)
  {
    return get_value<double>(VITAL_META_SLANT_RANGE, fid, slant_range);
  }

  virtual bool get_target_width(frame_id_t fid, double &target_width)
  {
    return get_value<double>(VITAL_META_TARGET_WIDTH, fid, target_width);
  }

  virtual bool get_platform_heading_angle(frame_id_t fid, double &heading)
  {
    return get_value<double>(VITAL_META_PLATFORM_HEADING_ANGLE, fid, heading);
  }

  virtual bool get_platform_pitch_angle(frame_id_t fid, double &pitch)
  {
    return get_value<double>(VITAL_META_PLATFORM_PITCH_ANGLE, fid, pitch);
  }

  virtual bool get_platform_roll_angle(frame_id_t fid, double &roll)
  {
    return get_value<double>(VITAL_META_PLATFORM_ROLL_ANGLE, fid, roll);
  }

  virtual bool get_sensor_rel_az_angle(frame_id_t fid, double &rel_az_angle)
  {
    return get_value<double>(VITAL_META_SENSOR_REL_AZ_ANGLE, fid, rel_az_angle);
  }

  virtual bool get_sensor_rel_el_angle(frame_id_t fid, double &rel_el_angle)
  {
    return get_value<double>(VITAL_META_SENSOR_REL_EL_ANGLE, fid, rel_el_angle);
  }

  virtual bool get_sensor_rel_roll_angle(frame_id_t fid, double &rel_roll_angle)
  {
    return get_value<double>(VITAL_META_SENSOR_REL_ROLL_ANGLE, fid, rel_roll_angle);
  }

  virtual std::set<frame_id_t> frames()
  {
    std::set<frame_id_t> fids;
    for (auto &m : data_)
    {
      fids.insert(m.first);
    }
    return fids;
  }

protected:

  template<typename T>
  bool get_value(vital_metadata_tag tag, frame_id_t fid, T& val)
  {
    auto d_it = data_.find(fid);
    if (d_it == data_.end())
    {
      return false;
    }

    auto &mdv = d_it->second;
    for (auto md : mdv)
    {
      if (md->has(tag))
      {
        md->find(tag).data(val);
        return true;
      }
    }
    return false;
  }

  /// The map from integer IDs to metadata shared pointers
  map_metadata_t data_;
};

}} // end namespace vital

#endif // KWIVER_VITAL_METADATA_MAP_H_
