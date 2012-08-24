/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H
#define VISTK_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <string>
#include <vector>

/**
 * \file pipe_declaration_types.h
 *
 * \brief Types for the AST of a pipeline declaration.
 */

namespace vistk
{

/// The type for a token in the AST.
typedef std::string token_t;

/// The type for a flag on a configuration key.
typedef token_t config_flag_t;
/// The type for a collection of flags on a configuration key.
typedef std::vector<config_flag_t> config_flags_t;

/// The type for a configuration provider.
typedef token_t config_provider_t;

/**
 * \struct config_key_options_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief Options for a configuration key.
 */
struct config_key_options_t
{
  /// Flags on the configuration (if requested).
  boost::optional<config_flags_t> flags;
  /// The configuration provider (if requested).
  boost::optional<config_provider_t> provider;
};

/**
 * \struct config_key_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for information on a configuration key.
 */
struct config_key_t
{
  /// The configuration path.
  config::keys_t key_path;
  /// Options for the key.
  config_key_options_t options;
};

/**
 * \struct config_value_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a configuration value.
 */
struct config_value_t
{
  /// The key for the configuration.
  config_key_t key;
  /// The value of the configuration.
  config::value_t value;
};

/// The type for a collection of configuration values.
typedef std::vector<config_value_t> config_values_t;

/**
 * \struct map_options_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief Options for a mapping connection.
 */
struct map_options_t
{
  /// Flags for the mapping.
  boost::optional<process::port_flags_t> flags;
};

/**
 * \struct group_input_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a group input mapping.
 */
struct group_input_t
{
  /// Options for the mapping.
  map_options_t options;
  /// The name of the group input port.
  process::port_t from;
  /// The address of the mapped downstream port.
  process::port_addr_t to;
};

/**
 * \struct group_output_t pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a group output mapping.
 */
struct group_output_t
{
  /// Options for the mapping.
  map_options_t options;
  /// The address of the mapped upstream port.
  process::port_addr_t from;
  /// The name of the group output port.
  process::port_t to;
};

/**
 * \struct config_pipe_block pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a configuration block.
 */
struct config_pipe_block
{
  /// The common path of the configuration.
  config::keys_t key;
  /// The values for the configuration block.
  config_values_t values;
};

/**
 * \struct process_pipe_block pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a process block.
 */
struct process_pipe_block
{
  /// The name of the process.
  process::name_t name;
  /// The type of the process.
  process::type_t type;
  /// Associated configuration values.
  config_values_t config_values;
};

/**
 * \struct connect_pipe_block pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a connection block.
 */
struct connect_pipe_block
{
  /// The address of the upstream port.
  process::port_addr_t from;
  /// The address of the downstream port.
  process::port_addr_t to;
};

/// A variant over the possible blocks that may be contained within a group.
typedef boost::variant<config_value_t, group_input_t, group_output_t> group_subblock_t;
/// A type for a collection of group subblocks.
typedef std::vector<group_subblock_t> group_subblocks_t;

/**
 * \struct group_pipe_block pipe_declaration_types.h <vistk/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a group block.
 */
struct group_pipe_block
{
  /// The name of the group.
  process::name_t name;
  /// Subblocks of the group.
  group_subblocks_t subblocks;
};

/// A discriminating union over all available block types.
typedef boost::variant
  < config_pipe_block
  , process_pipe_block
  , connect_pipe_block
  , group_pipe_block
  > pipe_block;

/// A type for a collection of pipe blocks.
typedef std::vector<pipe_block> pipe_blocks;

}

#endif // VISTK_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H
