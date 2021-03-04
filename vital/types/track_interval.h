// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for \link kwiver::vital::track_interval type
 */

#ifndef VITAL_TRACK_INTERVAL_H_
#define VITAL_TRACK_INTERVAL_H_

#include <vital/types/timestamp.h>

#include <vital/vital_types.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
struct track_interval
{
  track_id_t track;
  timestamp start;
  timestamp stop;
};

} // namespace vital

} // namespace kwiver

#endif
