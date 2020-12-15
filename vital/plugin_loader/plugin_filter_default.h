// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_FITAL_PLUGIN_FILTER_DEFAULT_H
#define KWIVER_FITAL_PLUGIN_FILTER_DEFAULT_H

#include <vital/plugin_loader/vital_vpm_export.h>

#include <vital/plugin_loader/plugin_loader_filter.h>

namespace kwiver {
namespace vital {

// -----------------------------------------------------------------
/** Default plugin loader filter.
 *
 * This filter excludes duplicate plugins. An exception is thrown if a
 * duplicate is found.
 */
class VITAL_VPM_EXPORT plugin_filter_default
  : public plugin_loader_filter
{
public:
  // -- CONSTRUCTORS --
  plugin_filter_default() = default;
  virtual ~plugin_filter_default() = default;

  virtual bool add_factory( plugin_factory_handle_t fact ) const;

}; // end class plugin_filter_default

} } // end namespace

#endif // KWIVER_FITAL_PLUGIN_FILTER_DEFAULT_H
