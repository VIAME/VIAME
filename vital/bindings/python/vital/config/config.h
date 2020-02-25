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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#ifndef KWIVER_VITAL_PYTHON_CONFIG_H
#define KWIVER_VITAL_PYTHON_CONFIG_H

#include <vital/config/config_block.h>
#include <vital/config/config_difference.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kwiver {
namespace vital {
namespace python {

  /// Cast python object to a configuration block value
  kwiver::vital::config_block_value_t
  config_block_set_value_cast( py::object const& value );

  /// Set a value in configuration block using key and value
  void
  config_set_value( kwiver::vital::config_block_sptr self,
                    kwiver::vital::config_block_key_t const& key,
                    kwiver::vital::config_block_key_t const& value  );

  /// Get a value in a configuration block using key
  kwiver::vital::config_block_value_t
  config_get_value( kwiver::vital::config_block_sptr self,
                    kwiver::vital::config_block_key_t const& key );

  /// Get a value in a configuration block using key and a default value
  kwiver::vital::config_block_value_t
  config_get_value_with_default( kwiver::vital::config_block_sptr self,
                                 kwiver::vital::config_block_key_t const& key,
                                 kwiver::vital::config_block_value_t const& def );

  /// Determine the number of elements in config block
  py::size_t
  config_len( kwiver::vital::config_block_sptr self );

  /// Get a value in a configuration block using key
  kwiver::vital::config_block_value_t
  config_getitem( kwiver::vital::config_block_sptr self,
                  kwiver::vital::config_block_key_t const& key );

  /// Set a value in a configuration block using key and value
  void
  config_setitem( kwiver::vital::config_block_sptr self,
                  kwiver::vital::config_block_key_t const& key,
                  py::object const& value );

  /// Delete value in a configuration block using a key
  void
  config_delitem( kwiver::vital::config_block_sptr self,
                  kwiver::vital::config_block_key_t const&  key );

}
}
}

/// pybind11 module for config
void
config( py::module& m );
#endif
