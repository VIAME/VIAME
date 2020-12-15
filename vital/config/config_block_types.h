// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

/// Shared pointer for the \c config_block class
typedef std::shared_ptr<config_block> config_block_sptr;

/// The type to be used for file and directory paths
typedef std::string config_path_t;

typedef std::vector< std::string > config_path_list_t;

} }

#endif
