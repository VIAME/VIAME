// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_KWIVER_H
#define INCL_TRACK_KWIVER_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kwiver/track_kwiver_export.h>

#include <track_oracle/core/track_base.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KWIVER_EXPORT track_kwiver_type: public track_base< track_kwiver_type >
{
  // The KWIVER format is special; it has no predefined types.
  // However, the file format manager requires an instance of
  // some representation of the schema.
};

} // ...track_oracle
} // ...kwiver

#endif
