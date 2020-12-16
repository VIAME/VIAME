// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_DESCRIPTOR_MOTION_H
#define INCL_DESCRIPTOR_MOTION_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <iostream>

namespace kwiver {
namespace track_oracle {

struct VIBRANT_DESCRIPTORS_EXPORT descriptor_motion_type
{
  double ground_pos_x;
  double ground_pos_y;
  double ground_speed;
  double ground_acceleration;
  double heading;
  double delta_heading;
  double exp_heading;
  double ang_momentum;
  double curvature;

  descriptor_motion_type(void)
    : ground_pos_x(0.0),
      ground_pos_y(0.0),
      ground_speed(0.0),
      ground_acceleration(0.0),
      heading(0.0),
      delta_heading(0.0),
      exp_heading(0.0),
      ang_momentum(0.0),
      curvature(0.0)
  { }

  bool operator==(const descriptor_motion_type& a) const
  {
    return (this->ground_pos_x == a.ground_pos_x) &&
      (this->ground_pos_y == a.ground_pos_y) &&
      (this->ground_speed == a.ground_speed) &&
      (this->ground_acceleration == a.ground_acceleration) &&
      (this->heading == a.heading) &&
      (this->delta_heading == a.delta_heading) &&
      (this->exp_heading == a.exp_heading) &&
      (this->ang_momentum == a.ang_momentum) &&
      (this->curvature == a.curvature);
  }

};

VIBRANT_DESCRIPTORS_EXPORT std::ostream& operator<<( std::ostream& os, const descriptor_motion_type& );
VIBRANT_DESCRIPTORS_EXPORT std::istream& operator>>( std::istream& is, descriptor_motion_type& );

} // ...track_oracle
} // ...kwiver

#endif /* INCL_DESCRIPTOR_MOTION_H */
