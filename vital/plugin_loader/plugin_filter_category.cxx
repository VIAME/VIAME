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
