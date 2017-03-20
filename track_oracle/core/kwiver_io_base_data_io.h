/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_KWIVER_IO_BASE_DATA_IO_H
#define INCL_KWIVER_IO_BASE_DATA_IO_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <iostream>
#include <string>
#include <utility>
#include <set>

namespace kwiver {
namespace track_oracle {

TRACK_ORACLE_EXPORT std::ostream& operator<<(std::ostream& os, const std::set< std::string >& v );
TRACK_ORACLE_EXPORT std::istream& operator>>(std::istream& is, std::set< std::string >& v );
TRACK_ORACLE_EXPORT std::ostream& operator<<(std::ostream& os, const std::pair<unsigned, unsigned >& v );
TRACK_ORACLE_EXPORT std::istream& operator>>(std::istream& is, std::pair<unsigned, unsigned>& v );

} // ...track_oracle
} // ...kwiver

#endif
