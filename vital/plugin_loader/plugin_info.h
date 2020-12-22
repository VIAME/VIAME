// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_PLUGIN_LOADER_PLUGIN_INFO_H
#define VITAL_PLUGIN_LOADER_PLUGIN_INFO_H

#define PLUGIN_INFO(NAME, DESCRIP)             \
  static constexpr char const* _plugin_name{ NAME };            \
  static constexpr char const* _plugin_description{ DESCRIP };

#endif // VITAL_PLUGIN_LOADER_PLUGIN_INFO_H
