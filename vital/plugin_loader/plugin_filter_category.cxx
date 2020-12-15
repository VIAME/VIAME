// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "plugin_filter_category.h"
#include "plugin_factory.h"
#include "plugin_loader.h"

#include <iostream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
plugin_filter_category
::plugin_filter_category( plugin_filter_category::condition cond,
                          const std::string& cat)
  : m_condition( cond ),
    m_category( cat )
{ }

// ------------------------------------------------------------------
/**
 * @brief Filter select or reject plugins by category.
 *
 * This method compares the factory against the supplied category and,
 * depending on whether the category is included or excluded, returns
 * that indication.
 *
 * @param fact Factory object handle
 *
 * @return \b true if factory is to be added; \b false if factory
 * should not be added.
 *
 */
bool
plugin_filter_category
::add_factory( plugin_factory_handle_t fact ) const
{
  std::string cat;
  if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, cat ) )
  {
    if ( m_condition == condition::EQUAL )
    {
      return (cat == m_category );
    }
    else
    {
      return (cat != m_category );
    }
  }

  // Select if category not available
  return true;

}

} } // end namespace
