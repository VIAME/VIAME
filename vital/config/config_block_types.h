/*ckwg +29
 * Copyright 2013-2014 by Kitware, Inc.
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

#ifndef KWIVER_CONFIG_BLOCK_TYPES_H_
#define KWIVER_CONFIG_BLOCK_TYPES_H_

#include <string>
#include <memory>
#include <vector>

//
// Define config block supporting types
//

namespace kwiver {
namespace vital {

class config_block;

/// The type that represents a configuration value key.
typedef std::string config_block_key_t;

/// The type that represents a collection of configuration keys.
typedef std::vector<config_block_key_t> config_block_keys_t;

/// The type that represents a stored configuration value.
typedef std::string config_block_value_t;

/// The type that represents a description of a configuration key.
typedef std::string config_block_description_t;

class config_block;
/// Shared pointer for the \c config_block class
typedef std::shared_ptr<config_block> config_block_sptr;

/// The type to be used for file and directory paths
typedef std::string config_path_t;

typedef std::vector< std::string > config_path_list_t;

} }

#endif /* KWIVER_CONFIG_BLOCK_TYPES_H_ */
