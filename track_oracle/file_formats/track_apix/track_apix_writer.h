// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_WRITER_APIX_H
#define INCL_TRACK_WRITER_APIX_H

#include <vital/vital_config.h>
#include <track_oracle/track_apix/track_apix_export.h>

#include <string>
#include <track_oracle/track_oracle_core.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_APIX_EXPORT track_apix_writer
{

// only one track per shapefile

static
bool
write( track_handle_type trk, const std::string& filename, const std::string time_format_str = "%d-%02d-%02d T %02d:%02d:%02d.%03dZ" );

};

} // ...track_oracle
} // ...kwiver

#endif
