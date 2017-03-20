/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_SCHEMA_FACTORY_H
#define INCL_SCHEMA_FACTORY_H

//
// Construct a schema, currently by cloning fields out of known
// file formats (hence the coupling with file_format_manager rather
// than with the track_oracle core.)
//

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_oracle_file_formats_export.h>

#include <string>

namespace kwiver {
namespace track_oracle {

class track_base_impl;

namespace schema_factory {

bool TRACK_ORACLE_FILE_FORMATS_EXPORT
clone_field_into_schema( track_base_impl& schema,
                         const std::string& name );

} // ...schema_factory

} // ...track_oracle
} // ...kwiver

#endif
