// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_MODULE_HELPERS_H
#define KWIVER_VITAL_PYTHON_MODULE_HELPERS_H

#include <string>

bool is_python_library_loaded( const std::string& python_library_path );
bool load_python_library_from_env();
bool load_python_library_from_interpretor(
  const std::string python_library_path );

std::string find_python_library();

/// Ensure an embedded Python interpreter is available.
///
/// Initializes an interpreter if one is not already running.  Returns
/// \c false (without aborting the process) if no interpreter could be
/// initialized, for example when the host executable has no Python
/// environment configured; callers should then skip Python plugin loading.
bool check_and_initialize_python_interpretor();

#endif
