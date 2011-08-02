/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"

#include "pipe_declaration_types.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant.hpp>

#include <utility>
#include <vector>

/**
 * \file pipe_bakery.cxx
 *
 * \brief Implementation of baking a pipeline.
 */

namespace vistk
{

class VISTK_PIPELINE_UTIL_NO_EXPORT pipe_bakery
  : public boost::static_visitor<>
{
  public:
    pipe_bakery();
    ~pipe_bakery();

    void operator () (config_pipe_block const& config_block);
    void operator () (process_pipe_block const& process_block);
    void operator () (connect_pipe_block const& connect_block);
    void operator () (group_pipe_block const& group_block);

    /**
     * \note We do *not* want std::map for the groupings. With a map, we may
     * hide errors in the blocks (setting ro values, duplicate process names,
     * etc.)
     */

    typedef std::pair<config_provider_t, config::value_t> provider_request_t;
    typedef boost::variant<config::value_t, provider_request_t> config_reference_t;
    typedef boost::tuple<config_reference_t, bool> config_info_t;
    typedef std::pair<config::key_t, config_info_t> config_decl_t;
    typedef std::vector<config_decl_t> config_decls_t;

    typedef std::pair<process::name_t, process_registry::type_t> process_decl_t;
    typedef std::vector<process_decl_t> process_decls_t;

    typedef std::pair<process::port_addr_t, process::port_addr_t> connection_t;
    typedef std::vector<connection_t> connections_t;

    typedef boost::tuple<process::port_t, process::port_flags_t, process::port_addr_t> mapping_t;
    typedef std::vector<mapping_t> mappings_t;
    typedef std::pair<mappings_t, mappings_t> group_info_t;
    typedef std::pair<process::name_t, group_info_t> group_decl_t;
    typedef std::vector<group_decl_t> group_decls_t;

    static config::key_t flatten_keys(config::keys_t const& keys);
    void register_config_value(config::key_t const& root_key, config_value_t const& value);

    config_decls_t m_configs;
    process_decls_t m_processes;
    connections_t m_connections;
    group_decls_t m_groups;
};

pipeline_t
bake_pipe_blocks(pipe_blocks const& /*blocks*/)
{
  pipeline_t pipeline;

  /// \todo Bake pipe blocks into a pipeline.

  return pipeline;
}

pipe_bakery
::pipe_bakery()
{
}

pipe_bakery
::~pipe_bakery()
{
}

void
pipe_bakery
::operator () (config_pipe_block const& config_block)
{
  /// \todo Implement.
}

void
pipe_bakery
::operator () (process_pipe_block const& process_block)
{
  /// \todo Implement.
}

void
pipe_bakery
::operator () (connect_pipe_block const& connect_block)
{
  /// \todo Implement.
}

void
pipe_bakery
::operator () (group_pipe_block const& group_block)
{
  /// \todo Implement.
}

void
pipe_bakery
::register_config_value(config::key_t const& root_key, config_value_t const& value)
{
  /// \todo Implement.
}

config::key_t
pipe_bakery
::flatten_keys(config::keys_t const& keys)
{
  return boost::join(keys, config::block_sep);
}

}
