// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_DESCRIPTOR_METADATA_H
#define INCL_DESCRIPTOR_METADATA_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <iostream>

namespace kwiver {
namespace track_oracle {

struct VIBRANT_DESCRIPTORS_EXPORT descriptor_metadata_type
{
  double gsd;
  double sensor_latitude;
  double sensor_longitude;
  double upper_left_corner_latitude;
  double upper_left_corner_longitude;
  double upper_right_corner_latitude;
  double upper_right_corner_longitude;
  double lower_left_corner_latitude;
  double lower_left_corner_longitude;
  double lower_right_corner_latitude;
  double lower_right_corner_longitude;
  double horizontal_field_of_view;
  double vertical_field_of_view;
  long long timestamp_microseconds_since_1970;
  float slant_range;

  descriptor_metadata_type(void)
    : gsd(0.0),
      sensor_latitude(0.0),
      sensor_longitude(0.0),
      upper_left_corner_latitude(0.0),
      upper_left_corner_longitude(0.0),
      upper_right_corner_latitude(0.0),
      upper_right_corner_longitude(0.0),
      lower_left_corner_latitude(0.0),
      lower_left_corner_longitude(0.0),
      lower_right_corner_latitude(0.0),
      lower_right_corner_longitude(0.0),
      horizontal_field_of_view(0.0),
      vertical_field_of_view(0.0),
      timestamp_microseconds_since_1970(0),
      slant_range(0.0f)
  { }

  bool operator==(const descriptor_metadata_type& a) const
  {
    return (this->gsd == a.gsd) &&
      (this->sensor_latitude == a.sensor_latitude) &&
      (this->sensor_longitude == a.sensor_longitude) &&
      (this->upper_left_corner_latitude == a.upper_left_corner_latitude) &&
      (this->upper_left_corner_longitude == a.upper_left_corner_longitude) &&
      (this->upper_right_corner_latitude == a.upper_right_corner_latitude) &&
      (this->upper_right_corner_longitude == a.upper_right_corner_longitude) &&
      (this->lower_left_corner_latitude == a.lower_left_corner_latitude) &&
      (this->lower_left_corner_longitude == a.lower_left_corner_longitude) &&
      (this->lower_right_corner_latitude == a.lower_right_corner_latitude) &&
      (this->lower_right_corner_longitude == a.lower_right_corner_longitude) &&
      (this->horizontal_field_of_view == a.horizontal_field_of_view) &&
      (this->vertical_field_of_view == a.vertical_field_of_view) &&
      (this->timestamp_microseconds_since_1970 == a.timestamp_microseconds_since_1970) &&
      (this->slant_range == a.slant_range);
  }

};

VIBRANT_DESCRIPTORS_EXPORT std::ostream& operator<<( std::ostream& os, const descriptor_metadata_type& );
VIBRANT_DESCRIPTORS_EXPORT std::istream& operator>>( std::istream& is, descriptor_metadata_type& );

} // ...track_oracle
} // ...kwiver

#endif /* INCL_DESCRIPTOR_METADATA_H */
