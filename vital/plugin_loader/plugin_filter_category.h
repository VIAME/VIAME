// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_FITAL_PLUGIN_FILTER_CATEGORY_H
#define KWIVER_FITAL_PLUGIN_FILTER_CATEGORY_H

#include <vital/plugin_loader/vital_vpm_export.h>

#include <vital/plugin_loader/plugin_loader_filter.h>

namespace kwiver {
namespace vital {

// -----------------------------------------------------------------
/** Select plugin based on category name.
 *
 * This filter class selects a plugin based on the specified category
 * name and condition. Plugins of a specific category can be included
 * or excluded from loading.
 *
 * EQUAL selects or includes plugins of specified category.
 * NOT_EQUAL excludes plugins of specified category.
 */
class VITAL_VPM_EXPORT plugin_filter_category
  : public plugin_loader_filter
{
public:
  enum class condition { EQUAL, // select plugins of specified category
                         NOT_EQUAL }; // excludeplugins of specified category

  // -- CONSTRUCTORS --
  plugin_filter_category( plugin_filter_category::condition cond,
                          const std::string& cat);
  virtual ~plugin_filter_category() = default;

  virtual bool add_factory( plugin_factory_handle_t fact ) const;

private:
  plugin_filter_category::condition m_condition;
  std::string m_category;

}; // end class plugin_filter_category

} } // end namespace

#endif // KWIVER_FITAL_PLUGIN_FILTER_CATEGORY_H
