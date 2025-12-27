/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_FILENAME_TO_TIMESTAMP_H
#define VIAME_CORE_FILENAME_TO_TIMESTAMP_H

#include <plugins/core/viame_core_export.h>

#include <vital/types/timestamp.h>

namespace viame
{

/// Convert a filename in many different common formats to a kwiver timestamp
///
/// @param filename input filename
/// @param auto_discover attempt to automatically detect never-seen formats
///
/// @throws runtime_error on invalid or unable to parse filename format
///
VIAME_CORE_EXPORT
kwiver::vital::time_usec_t
convert_to_timestamp( const std::string& filename,
                      const bool auto_discover = false );

} // end namespace viame

#endif // VIAME_CORE_FILENAME_TO_TIMESTAMP_H
