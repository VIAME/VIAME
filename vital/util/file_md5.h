// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_UTIL_FILE_MD5_H
#define KWIVER_VITAL_UTIL_FILE_MD5_H

#include <vital/util/vital_util_export.h>

#include <string>

namespace kwiver {
namespace vital {

/**
 * @brief Compute the md5 of a file
 *
 * Compute the md5 sum of the named file
 *
 * @param fn Path to the file
 *
 * @return MD5 sum as a string; empty if an error ocurred.
 */

VITAL_UTIL_EXPORT std::string
file_md5( const std::string& fn );

} // ...vital
} // ...kwiver

#endif
