/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
