// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for \link viame::track_interval type

#ifndef VITAL_TRACK_INTERVAL_H_
#define VITAL_TRACK_INTERVAL_H_

#include <viame/core_types/timestamp.h>

#include <viame/core_types/viame_core_types.h>

namespace viame {

// ----------------------------------------------------------------------------
struct track_interval
{
  track_id_t track;
  timestamp start;
  timestamp stop;
};

} // namespace viame

#endif
