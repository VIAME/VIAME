/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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
