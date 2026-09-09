// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Wrapper over C functions to get executable path and module path.

#ifndef KWIVER_GET_PATHS_H
#define KWIVER_GET_PATHS_H

#include <viame/algorithm_framework/util/vital_util_export.h>
#include <viame/algorithm_framework/vital_config.h>

#include <string>

namespace kwiver {

namespace vital {

/// @brief Get path to current executable.
///
/// Get the name of the directory that contains the current executable
/// file. The returned string does not include the file name.
///
/// @return Directory name.
std::string VITAL_UTIL_EXPORT get_executable_path();

/// @brief Get path to the current module.
///
/// @return Directory name.
std::string VITAL_UTIL_EXPORT get_module_path();

} // namespace vital

} // namespace kwiver

#endif // KWIVER_GET_PATHS_H
