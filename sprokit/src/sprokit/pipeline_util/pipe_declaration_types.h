// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H
#define SPROKIT_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H

#include <vital/config/config_block.h>
#include <vital/util/source_location.h>
#include <vital/internal/variant/variant.hpp>

#include <sprokit/pipeline/process.h>

#include <string>
#include <vector>

/**
 * \file pipe_declaration_types.h
 *
 * \brief Types for the AST of a pipeline declaration.
 */

namespace sprokit {

/// The type for a token in the AST.
typedef std::string token_t;

/// The type for a flag on a configuration key.
typedef token_t config_flag_t;

/// The type for a collection of flags on a configuration key.
/*
 * These flags are represented as strings, which provides an easily
 * extensible way of managing these options. The definition of all
 * valid values should be more tightly bound to this type
 * definition. e.g. this should be an enum.
 */
typedef std::vector<config_flag_t> config_flags_t;

/// The type for a configuration provider.
typedef token_t config_provider_t;

// ------------------------------------------------------------------
/**
 * \struct config_value_t pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a configuration value.
 *
 * This struct is used to represent a set of config entries. This is
 * used to represent the config of the pipeline, and a process,
 *
 */
struct config_value_t
{
  /// The configuration path. Each element in the vector is a portion of the key.
  kwiver::vital::config_block_keys_t key_path;

  // Attributes that are associated with this config key
  // An empty list indicates no flags or attributes.
  config_flags_t flags;

  /// The value of the configuration.
  kwiver::vital::config_block_value_t value;

  /// Source location of definition
  kwiver::vital::source_location loc;
};

/// The type for a collection of configuration values.
typedef std::vector<config_value_t> config_values_t;

// ------------------------------------------------------------------
/**
 * \struct config_pipe_block pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a configuration block.
 *
 */
struct config_pipe_block
{
  /** The common path of the configuration.
   *
   * This entry contains the name of the config block which follows
   * the "config" keyword. These need to be kept separate so that this
   * internal representation can be used to recreate a valid pipe file.
   */
  kwiver::vital::config_block_keys_t key; // Name of the config block (after "config" keyword)

  /// The values for the configuration block.
  config_values_t values; // vector of key/value pairs

  /// Source location of "config" keyword
  kwiver::vital::source_location loc;
};

// ------------------------------------------------------------------
/**
 * \struct process_pipe_block pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a process block.
 *
 * This struct represents a process definition. It contains the
 * process class, the instance name of that process, and associated
 * configuration.
 */
struct process_pipe_block
{
  /// The name of the process.
  process::name_t name;

  /// The type of the process.
  process::type_t type;

  /// Associated configuration values.
  config_values_t config_values;

  /// Source location of "process" keyword
  kwiver::vital::source_location loc;
};

// ------------------------------------------------------------------
/**
 * \struct connect_pipe_block pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a connection block.
 */
struct connect_pipe_block
{
  /// The address of the upstream port.
  process::port_addr_t from;

  /// The address of the downstream port.
  process::port_addr_t to;

  /// Source location of definition
  kwiver::vital::source_location loc;

};

/// A discriminating union over all available pipeline block types.
typedef kwiver::vital::variant
  < config_pipe_block
  , process_pipe_block
  , connect_pipe_block
  > pipe_block;

/// A type for a collection of pipe blocks.
typedef std::vector<pipe_block> pipe_blocks;

// ------------------------------------------------------------------
/**
 * \struct cluster_config_t pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for cluster config.
 *
 * There is an element of this type for each config entry associated with the cluster.
 */
struct cluster_config_t
{
  /// Description of the configuration value.
  kwiver::vital::config_block_description_t description;

  /// The configuration declaration.
  config_value_t config_value;
};

// ------------------------------------------------------------------
/**
 * \struct cluster_input_t pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a cluster input mapping.
 */
struct cluster_input_t
{
  /// Description of the cluster's port.
  process::port_description_t description;

  /// The name of the cluster's input port.
  process::port_t from;

  /// The addresses of the mapped port.
  process::port_addrs_t targets;

  /// Source location of definition
  kwiver::vital::source_location loc;
};

// ------------------------------------------------------------------
/**
 * \struct cluster_output_t pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a cluster output mapping.
 */
struct cluster_output_t
{
  /// Description of the cluster's port.
  process::port_description_t description;

  /// The address of the mapped upstream port.
  process::port_addr_t from;

  /// The name of the cluster's output port.
  process::port_t to;

  // Add source location
  /// Source location of definition
  kwiver::vital::source_location loc;
};

/// A variant over the possible blocks that may be contained within a cluster.
typedef kwiver::vital::variant<cluster_config_t, cluster_input_t, cluster_output_t> cluster_subblock_t;

/// A type for a collection of cluster subblocks.
typedef std::vector<cluster_subblock_t> cluster_subblocks_t;

// ------------------------------------------------------------------
/**
 * \struct cluster_pipe_block pipe_declaration_types.h <sprokit/pipeline_util/pipe_declaration_types.h>
 *
 * \brief A structure for a cluster block.
 */
struct cluster_pipe_block
{
  /// The type (name) of the cluster.
  process::type_t type;

  /// The description of the cluster.
  process::description_t description;

  /// Subblocks of the cluster.
  cluster_subblocks_t subblocks;
};

/// A discriminating union over all available cluster block types.
typedef kwiver::vital::variant
  < config_pipe_block
  , process_pipe_block
  , connect_pipe_block
  , cluster_pipe_block
  > cluster_block;

/// A type for a collection of cluster blocks.
typedef std::vector<cluster_block> cluster_blocks;

}

#endif // SPROKIT_PIPELINE_UTIL_PIPE_DECLARATION_TYPES_H
