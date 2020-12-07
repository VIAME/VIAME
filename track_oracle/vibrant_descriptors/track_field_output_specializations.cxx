// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_field_output_specializations.h"

#include <stdexcept>
#include <utility>
#include <iostream>
#include <vector>
#include <sstream>

using std::ostream;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

namespace kwiver {
namespace track_oracle {

template< >
ostream& operator<<( ostream& os,
                         const track_field< descriptor_cutic_type >& f) {
  os << " (" << f.field_handle << ") " << f.name;
    try
  {
    descriptor_cutic_type d = f();

    //os << "[score_class=" << d.score_class << "] ";
    //os << "[score_type=" << d.score_type << "] ";
    //os << "[sim_temporal=" << d.sim_temporal.size()  << "] ";
    //os << "[desc_index=" << d.desc_index.size() << "] ";
    os << "[desc_raw size=" << d.desc_raw.size() << "] ";

    for (size_t i=0; i<d.desc_raw.size(); ++i)
    {
      os << d.desc_raw[i] << " ";
    }

  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< descriptor_metadata_type >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    descriptor_metadata_type d = f();
    os << "[gsd=" << d.gsd << "] ";
    os << "[sensor_latitude=" << d.sensor_latitude << "] ";
    os << "[sensor_longitude=" << d.sensor_longitude  << "] ";
    os << "[upper_left_corner_latitude=" << d.upper_left_corner_latitude << "] ";
    os << "[upper_left_corner_longitude=" << d.upper_left_corner_longitude << "] ";
    os << "[upper_right_corner_latitude=" << d.upper_right_corner_latitude << "] ";
    os << "[upper_right_corner_longitude=" << d.upper_right_corner_longitude << "] ";
    os << "[lower_left_corner_latitude=" << d.lower_left_corner_latitude << "] ";
    os << "[lower_left_corner_longitude=" << d.lower_left_corner_longitude << "] ";
    os << "[lower_right_corner_latitude=" << d.lower_right_corner_latitude << "] ";
    os << "[lower_right_corner_longitude=" << d.lower_right_corner_longitude << "] ";
    os << "[horizontal_field_of_view=" << d.horizontal_field_of_view << "] ";
    os << "[vertical_field_of_view=" << d.vertical_field_of_view << "] ";
    os << "[timestamp_microseconds_since_1970=" << d.timestamp_microseconds_since_1970 << "] ";
    os << "[slant_range=" << d.slant_range << "] ";
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< descriptor_motion_type >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    descriptor_motion_type d = f();
    os << "[ground_pos_x=" << d.ground_pos_x << "] ";
    os << "[ground_pos_y=" << d.ground_pos_y << "] ";
    os << "[ground_speed=" << d.ground_speed << "] ";
    os << "[ground_acceleration=" << d.ground_acceleration << "] ";
    os << "[heading" << d.heading << "] ";
    os << "[delta_heading=" << d.delta_heading << "] ";
    os << "[exp_heading=" << d.exp_heading << "] ";
    os << "[ang_momentum=" << d.ang_momentum << "] ";
    os << "[curvature=" << d.curvature << "] ";
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template<>
ostream& operator<<( ostream& os,
                         const track_field< descriptor_overlap_type >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    descriptor_overlap_type d = f();
    os << d;
  }
  catch (runtime_error const& )
  {
    os << " (no row set) ";
  }
  return os;
}

template<>
ostream& operator<<( ostream& os,
                         const track_field< descriptor_event_label_type >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    descriptor_event_label_type d = f();
    os << d;
  }
  catch (runtime_error const& )
  {
    os << " (no row set) ";
  }
  return os;
}

template<>
ostream& operator<<( ostream& os,
                         const track_field< descriptor_raw_1d_type >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    descriptor_raw_1d_type d = f();
    os << d;
  }
  catch (runtime_error const& )
  {
    os << " (no row set) ";
  }
  return os;
}

} // ...track_oracle
} // ...kwiver

