/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"

#include "pipe_declaration_types.h"
#include "providers.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

/**
 * \file pipe_bakery.cxx
 *
 * \brief Implementation of baking a pipeline.
 */

namespace vistk
{

namespace
{

static config::key_t const config_pipeline_key = config::key_t("_pipeline");

static config_flag_t const flag_read_only = config_flag_t("ro");

static config_provider_t const provider_config = config_provider_t("CONF");
static config_provider_t const provider_environment = config_provider_t("ENV");
static config_provider_t const provider_system = config_provider_t("SYS");

}

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

class VISTK_PIPELINE_UTIL_NO_EXPORT provider_dereferencer
  : public boost::static_visitor<pipe_bakery::config_reference_t>
{
  public:
    provider_dereferencer();
    provider_dereferencer(config_t const conf);
    ~provider_dereferencer();

    pipe_bakery::config_reference_t operator () (config::value_t const& value) const;
    pipe_bakery::config_reference_t operator () (pipe_bakery::provider_request_t const& request) const;
  private:
    typedef std::map<config_provider_t, provider_t> provider_map_t;
    provider_map_t m_providers;
};

class VISTK_PIPELINE_UTIL_NO_EXPORT ensure_provided
  : public boost::static_visitor<config::value_t>
{
  public:
    config::value_t operator () (config::value_t const& value) const;
    config::value_t operator () (pipe_bakery::provider_request_t const& request) const;
};

class VISTK_PIPELINE_UTIL_NO_EXPORT config_provider_sorter
  : public boost::static_visitor<>
{
  public:
    void operator () (config::key_t const& key, config::value_t const& value) const;
    void operator () (config::key_t const& key, pipe_bakery::provider_request_t const& request);

    config::keys_t sorted() const;
  private:
    struct vertex_name_t
    {
      typedef boost::vertex_property_tag kind;
    };
    typedef boost::property<vertex_name_t, config::key_t> name_property_t;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, name_property_t> config_graph_t;
    typedef boost::graph_traits<config_graph_t>::vertex_descriptor vertex_t;
    typedef std::vector<vertex_t> vertices_t;
    typedef std::map<config::key_t, vertex_t> vertex_map_t;

    vertex_map_t m_vertex_map;
    config_graph_t m_graph;
};

pipeline_t
bake_pipe_blocks(pipe_blocks const& blocks)
{
  pipeline_t pipe;

  pipe_bakery bakery;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(bakery));

  config_t global_conf = config::empty_config();

  // Build configuration.
  {
    // Dereference (non-configuration) providers.
    {
      BOOST_FOREACH (pipe_bakery::config_decl_t& decl, bakery.m_configs)
      {
        pipe_bakery::config_reference_t& ref = decl.second.get<0>();

        ref = boost::apply_visitor(provider_dereferencer(), ref);
      }
    }

    config_t conf = config::empty_config();

    BOOST_FOREACH (pipe_bakery::config_decl_t& decl, bakery.m_configs)
    {
      pipe_bakery::config_reference_t const& ref = decl.second.get<0>();

      config::key_t const key = decl.first;
      config::value_t val;

      // Only add provided configurations to the configuration.
      try
      {
        val = boost::apply_visitor(ensure_provided(), ref);
      }
      catch (...)
      {
        continue;
      }

      conf->set_value(key, val);
    }

    // Dereference configuration providers.
    {
      config_provider_sorter sorter;

      /// \bug Why must this be done?
      typedef boost::variant<config::key_t> dummy_variant;

      BOOST_FOREACH (pipe_bakery::config_decl_t& decl, bakery.m_configs)
      {
        pipe_bakery::config_reference_t const& ref = decl.second.get<0>();

        dummy_variant var(decl.first);

        boost::apply_visitor(sorter, var, ref);
      }

      config::keys_t keys = sorter.sorted();

      provider_dereferencer deref(conf);

      /// \todo This is algorithmically naive, but I'm not sure if there's a faster way.
      BOOST_FOREACH (config::key_t const& key, keys)
      {
        BOOST_FOREACH (pipe_bakery::config_decl_t& decl, bakery.m_configs)
        {
          config::key_t const& cur_key = decl.first;

          if (key != cur_key)
          {
            continue;
          }

          pipe_bakery::config_reference_t& ref = decl.second.get<0>();

          ref = boost::apply_visitor(deref, ref);

          config::value_t const val = boost::apply_visitor(ensure_provided(), ref);

          // Set the value in the intermediate configuration.
          conf->set_value(cur_key, val);
        }
      }
    }

    BOOST_FOREACH (pipe_bakery::config_decl_t& decl, bakery.m_configs)
    {
      pipe_bakery::config_reference_t const& ref = decl.second.get<0>();

      config::key_t const& key = decl.first;
      config::value_t const val = boost::apply_visitor(ensure_provided(), ref);

      global_conf->set_value(key, val);

      bool const is_readonly = decl.second.get<1>();

      if (is_readonly)
      {
        global_conf->mark_read_only(key);
      }
    }
  }

  // Create pipeline.
  config_t const pipeline_conf = global_conf->subblock_view(config_pipeline_key);

  pipe = pipeline_t(new pipeline(pipeline_conf));

  // Create processes.
  {
    process_registry_t reg = process_registry::self();

    BOOST_FOREACH (pipe_bakery::process_decl_t const& decl, bakery.m_processes)
    {
      process::name_t const& proc_name = decl.first;
      process_registry::type_t const& proc_type = decl.second;
      config_t const proc_conf = global_conf->subblock_view(proc_name);

      process_t proc = reg->create_process(proc_type, proc_conf);

      pipe->add_process(proc);
    }
  }

  // Create groups.
  {
    BOOST_FOREACH (pipe_bakery::group_decl_t const& decl, bakery.m_groups)
    {
      process::name_t const& group_name = decl.first;
      pipe_bakery::group_info_t const& group_info = decl.second;

      pipe->add_group(group_name);

      pipe_bakery::mappings_t const& input_mappings = group_info.first;

      BOOST_FOREACH (pipe_bakery::mapping_t const& mapping, input_mappings)
      {
        process::port_t const& port = mapping.get<0>();
        process::port_flags_t const& flags = mapping.get<1>();
        process::port_addr_t const& addr = mapping.get<2>();

        pipe->map_input_port(group_name, port, addr.first, addr.second, flags);
      }

      pipe_bakery::mappings_t const& output_mappings = group_info.second;

      BOOST_FOREACH (pipe_bakery::mapping_t const& mapping, output_mappings)
      {
        process::port_t const& port = mapping.get<0>();
        process::port_flags_t const& flags = mapping.get<1>();
        process::port_addr_t const& addr = mapping.get<2>();

        pipe->map_output_port(group_name, port, addr.first, addr.second, flags);
      }
    }
  }

  /// \todo Bake pipe blocks into a pipeline.

  return pipe;
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
  config::key_t const root_key = flatten_keys(config_block.key);

  config_values_t const& values = config_block.values;

  BOOST_FOREACH (config_value_t const& value, values)
  {
    register_config_value(root_key, value);
  }
}

void
pipe_bakery
::operator () (process_pipe_block const& process_block)
{
  config_values_t const& values = process_block.config_values;

  // Build the configuration value for the name of the process.
  config_value_t name_value;
  name_value.key.key_path.push_back(process::config_name);
  name_value.key.options.flags = config_flags_t();
  (*name_value.key.options.flags).push_back(flag_read_only);
  name_value.value = process_block.name;

  register_config_value(process_block.name, name_value);

  BOOST_FOREACH (config_value_t const& value, values)
  {
    register_config_value(process_block.name, value);
  }

  m_processes.push_back(process_decl_t(process_block.name, process_block.type));
}

void
pipe_bakery
::operator () (connect_pipe_block const& connect_block)
{
  m_connections.push_back(connection_t(connect_block.from, connect_block.to));
}

void
pipe_bakery
::operator () (group_pipe_block const& group_block)
{
  config_values_t const& values = group_block.config_values;

  BOOST_FOREACH (config_value_t const& value, values)
  {
    register_config_value(group_block.name, value);
  }

  process::port_flags_t default_flags;

  mappings_t input_mappings;

  BOOST_FOREACH (input_map_t const& map, group_block.input_mappings)
  {
    process::port_flags_t flags = default_flags;

    if (map.options.flags)
    {
      flags = *map.options.flags;
    }

    mapping_t const mapping = mapping_t(map.from, flags, map.to);

    input_mappings.push_back(mapping);
  }

  mappings_t output_mappings;

  BOOST_FOREACH (output_map_t const& map, group_block.output_mappings)
  {
    process::port_flags_t flags = default_flags;

    if (map.options.flags)
    {
      flags = *map.options.flags;
    }

    mapping_t const mapping = mapping_t(map.to, flags, map.from);

    output_mappings.push_back(mapping);
  }

  group_info_t const info = group_info_t(input_mappings, output_mappings);
  group_decl_t const decl = group_decl_t(group_block.name, info);

  m_groups.push_back(decl);
}

void
pipe_bakery
::register_config_value(config::key_t const& root_key, config_value_t const& value)
{
  config_key_t const key = value.key;

  config::key_t const subkey = flatten_keys(key.key_path);

  config_reference_t c_value;

  if (key.options.provider)
  {
    c_value = provider_request_t(*key.options.provider, value.value);
  }
  else
  {
    c_value = value.value;
  }

  bool is_readonly = false;

  if (key.options.flags)
  {
    BOOST_FOREACH (config_flag_t const& flag, *key.options.flags)
    {
      if (flag == flag_read_only)
      {
        is_readonly = true;
      }
      else
      {
        /// \todo Log warning about unrecognized flag.
      }
    }
  }

  config::key_t const full_key = root_key + config::block_sep + subkey;

  config_info_t const info = config_info_t(c_value, is_readonly);

  config_decl_t const decl = config_decl_t(full_key, info);

  m_configs.push_back(decl);
}

config::key_t
pipe_bakery
::flatten_keys(config::keys_t const& keys)
{
  return boost::join(keys, config::block_sep);
}

provider_dereferencer
::provider_dereferencer()
{
  m_providers[provider_system] = provider_t(new system_provider);
  m_providers[provider_environment] = provider_t(new environment_provider);
}

provider_dereferencer
::provider_dereferencer(config_t const conf)
{
  m_providers[provider_config] = provider_t(new config_provider(conf));
}

provider_dereferencer
::~provider_dereferencer()
{
}

pipe_bakery::config_reference_t
provider_dereferencer
::operator () (config::value_t const& value) const
{
  return value;
}

pipe_bakery::config_reference_t
provider_dereferencer
::operator () (pipe_bakery::provider_request_t const& request) const
{
  provider_map_t::const_iterator const i = m_providers.find(request.first);
  if (i == m_providers.end())
  {
    return request;
  }

  return (*i->second)(request.second);
}

config::value_t
ensure_provided
::operator () (config::value_t const& value) const
{
  return value;
}

config::value_t
ensure_provided
::operator () (pipe_bakery::provider_request_t const& /*request*/) const
{
  /// \todo Throw an exception.

  return config::value_t();
}

config::keys_t
config_provider_sorter
::sorted() const
{
  vertices_t vertices;

  try
  {
    boost::topological_sort(m_graph, std::back_inserter(vertices));
  }
  catch (boost::not_a_dag& e)
  {
    /// \todo Throw circular configuration provider exception.
  }

  config::keys_t keys;

  boost::property_map<config_graph_t, vertex_name_t>::const_type const key_prop = boost::get(vertex_name_t(), m_graph);

  BOOST_FOREACH (vertex_t const& vertex, vertices)
  {
    keys.push_back(boost::get(key_prop, vertex));
  }

  return keys;
}

void
config_provider_sorter
::operator () (config::key_t const& /*key*/, config::value_t const& /*value*/) const
{
}

void
config_provider_sorter
::operator () (config::key_t const& key, pipe_bakery::provider_request_t const& request)
{
  if (request.first != provider_config)
  {
    return;
  }

  boost::property_map<config_graph_t, vertex_name_t>::type key_prop = boost::get(vertex_name_t(), m_graph);

  typedef std::pair<vertex_map_t::iterator, bool> insertion_t;

  insertion_t from_iter = m_vertex_map.insert(std::make_pair(key, vertex_t()));
  insertion_t to_iter = m_vertex_map.insert(std::make_pair(request.second, vertex_t()));

  vertex_t s;
  vertex_t t;

  if (from_iter.second)
  {
    s = boost::add_vertex(m_graph);
    key_prop[s] = key;
    from_iter.first->second = s;
  }

  s = from_iter.first->second;

  if (to_iter.second)
  {
    t = boost::add_vertex(m_graph);
    key_prop[s] = request.second;
    to_iter.first->second = t;
  }

  t = to_iter.first->second;

  boost::add_edge(s, t, m_graph);
}

}
