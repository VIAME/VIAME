/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_CSV_H
#define INCL_TRACK_CSV_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_csv/track_csv_export.h>

#include <track_oracle/core/track_base.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_CSV_EXPORT track_csv_type: public track_base< track_csv_type >
{
  // The CSV format is special; it has no predefined types.
  // However, the file format manager requires an instance
  // of some representation of the schema.
};

} // ...track_oracle
} // ...kwiver

#endif
