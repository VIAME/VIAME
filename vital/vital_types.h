// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Shared type declarations for the VITAL module.
 *
 * This file contains the canonical type names for KWIVER-VITAL types.
 */

#ifndef KWIVER_CORE_TYPES_H
#define KWIVER_CORE_TYPES_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace kwiver {
namespace vital {

/// The type to be used for general strings
typedef std::string string_t;

/// The type to be used for file and directory paths
typedef std::string path_t;
typedef std::vector< path_t > path_list_t;

/// The type of a landmark ID number
typedef int64_t landmark_id_t;

/// The type of a track ID number
typedef int64_t track_id_t;

/// The type of a frame number
typedef int64_t frame_id_t;

// Time in micro-seconds
typedef int64_t time_usec_t;

// -- concrete types --
typedef double gsd_t;

// a short name for unsigned char
typedef unsigned char byte;

enum class clone_type
{
  SHALLOW,
  DEEP,
};

// Logger handle
class kwiver_logger;
using logger_handle_t = std::shared_ptr< kwiver_logger >;

/// The type of an activity ID number
typedef int64_t activity_id_t;

/// The type of an activity name
typedef std::string activity_label_t;

/// Global activity used to denote an undefined activity label
const activity_label_t UNDEFINED_ACTIVITY("UNDEFINED_ACTIVITY");

} } // end namespace

#endif // KWIVER_CORE_TYPES_H
