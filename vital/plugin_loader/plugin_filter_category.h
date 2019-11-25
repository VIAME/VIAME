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
