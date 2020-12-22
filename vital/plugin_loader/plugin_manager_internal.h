// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_PLUGIN_MANAGER_INTERNAL_H
#define VITAL_PLUGIN_MANAGER_INTERNAL_H

#include <vital/plugin_loader/plugin_manager.h>
namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/*
 * This class just exposes the protected members of the base class.
 * ---- For internal use only ----
 */
class plugin_manager_internal
  : public plugin_manager
{
public:
  static plugin_manager_internal& instance()
  {
    plugin_manager_internal* pm =
      reinterpret_cast< plugin_manager_internal* >(&plugin_manager::instance() );
    return *pm;
  }

  plugin_map_t const& plugin_map() { return plugin_manager::plugin_map(); }
  std::map< std::string, std::string > const& module_map() const { return plugin_manager::module_map(); }
  path_list_t const& search_path() const { return plugin_manager::search_path(); }
  plugin_loader* get_loader() { return plugin_manager::get_loader(); }
  std::vector< std::string > file_list() { return plugin_manager::file_list(); }

};

} } // end namespace

#endif // VITAL_PLUGIN_MANAGER_INTERNAL_H
