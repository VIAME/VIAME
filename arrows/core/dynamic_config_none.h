/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Header defining the interface to dynamic_config_none
 */

#ifndef ARROWS_CORE_DYNAMIC_CONFIG_NONE_H
#define ARROWS_CORE_DYNAMIC_CONFIG_NONE_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/dynamic_configuration.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for bypassing image conversion
class KWIVER_ALGO_CORE_EXPORT dynamic_config_none
  : public vital::algorithm_impl<dynamic_config_none, vital::algo::dynamic_configuration>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "none";

  /// Description of the algorithm
  static constexpr char const* description =
    "Null implementation of dynamic_configuration.\n\n"
    "This algorithm always returns an empty configuration block.";

  /// Default constructor
  dynamic_config_none();

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Return dynamic configuration values
  /**
   * This method returns dynamic configuration values. A valid config
   * block is returned even if there are not values being returned.
   */
  virtual kwiver::vital::config_block_sptr get_dynamic_configuration();
};

} } } // end namespace

#endif /* ARROWS_CORE_DYNAMIC_CONFIG_NONE_H */
