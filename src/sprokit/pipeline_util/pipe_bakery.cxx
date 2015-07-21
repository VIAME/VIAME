/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include "pipe_bakery.h"
#include "pipe_bakery_exception.h"

#include "load_pipe.h"
#include "pipe_declaration_types.h"
#include "providers.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_registry.h>

#if __cplusplus < 201103L
#include <boost/version.hpp>
// This header appeared in 1.50.0, but was buggy.
// XXX(boost): 1.52.0
#if BOOST_VERSION >= 105200
#include <boost/algorithm/cxx11/copy_if.hpp>
#endif
#endif
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <istream>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

/**
 * \file pipe_bakery.cxx
 *
 * \brief Implementation of baking a pipeline.
 */

namespace sprokit
{

namespace
{

static kwiver::vital::config_block_key_t const config_pipeline_key = kwiver::vital::config_block_key_t("_pipeline");

static config_flag_t const flag_read_only = config_flag_t("ro");
static config_flag_t const flag_append = config_flag_t("append");
static config_flag_t const flag_append_prefix = config_flag_t("append=");
static config_flag_t const flag_append_comma = config_flag_t("comma");
static config_flag_t const flag_append_space = config_flag_t("space");
static config_flag_t const flag_append_path = config_flag_t("path");
static config_flag_t const flag_tunable = config_flag_t("tunable");

static config_provider_t const provider_config = config_provider_t("CONF");
static config_provider_t const provider_environment = config_provider_t("ENV");
static config_provider_t const provider_system = config_provider_t("SYS");

}

class bakery_base
  : public boost::static_visitor<>
{
  public:
    bakery_base();
    virtual ~bakery_base();

    void operator () (config_pipe_block const& config_block);
    void operator () (process_pipe_block const& process_block);
    void operator () (connect_pipe_block const& connect_block);

    /**
     * \note We do *not* want std::map for the block management. With a map, we
     * may hide errors in the blocks (setting ro values, duplicate process
     * names, etc.)
     */

    typedef std::pair<config_provider_t, kwiver::vital::config_block_value_t> provider_request_t;
    typedef boost::variant<kwiver::vital::config_block_value_t, provider_request_t> config_reference_t;
    class config_info_t
    {
      public:
        typedef enum
        {
          append_none,
          append_string,
          append_comma,
          append_space,
          append_path
        } append_t;

        config_info_t(config_reference_t const& ref,
                      bool ro,
                      append_t app);
        ~config_info_t();

        config_reference_t reference;
        bool read_only;
        append_t append;
    };
    typedef std::pair<kwiver::vital::config_block_key_t, config_info_t> config_decl_t;
    typedef std::vector<config_decl_t> config_decls_t;

    typedef std::pair<process::name_t, process::type_t> process_decl_t;
    typedef std::vector<process_decl_t> process_decls_t;

    config_decls_t m_configs;
    process_decls_t m_processes;
    process::connections_t m_connections;
  protected:
    void register_config_value(kwiver::vital::config_block_key_t const& root_key, config_value_t const& value);
};

pipeline_t
bake_pipe_from_file(path_t const& fname)
{
  return bake_pipe_blocks(load_pipe_blocks_from_file(fname));
}

pipeline_t
bake_pipe(std::istream& istr, path_t const& inc_root)
{
  return bake_pipe_blocks(load_pipe_blocks(istr, inc_root));
}

class pipe_bakery
  : public bakery_base
{
  public:
    pipe_bakery();
    ~pipe_bakery();

    using bakery_base::operator ();
};

static kwiver::vital::config_block_sptr extract_configuration_from_decls(bakery_base::config_decls_t& configs);

pipeline_t
bake_pipe_blocks(pipe_blocks const& blocks)
{
  pipeline_t pipe;

  pipe_bakery bakery;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(bakery));

  bakery_base::config_decls_t& configs = bakery.m_configs;
  kwiver::vital::config_block_sptr global_conf = extract_configuration_from_decls(configs);

  // Create pipeline.
  kwiver::vital::config_block_sptr const pipeline_conf = global_conf->subblock_view(config_pipeline_key);

  pipe = boost::make_shared<pipeline>(pipeline_conf);

  // Create processes.
  {
    process_registry_t reg = process_registry::self();

    BOOST_FOREACH (bakery_base::process_decl_t const& decl, bakery.m_processes)
    {
      process::name_t const& proc_name = decl.first;
      process::type_t const& proc_type = decl.second;
      kwiver::vital::config_block_sptr const proc_conf = global_conf->subblock_view(proc_name);

      process_t const proc = reg->create_process(proc_type, proc_name, proc_conf);

      pipe->add_process(proc);
    }
  }

  // Make connections.
  {
    BOOST_FOREACH (process::connection_t const& conn, bakery.m_connections)
    {
      process::port_addr_t const& up = conn.first;
      process::port_addr_t const& down = conn.second;

      process::name_t const& up_name = up.first;
      process::port_t const& up_port = up.second;
      process::name_t const& down_name = down.first;
      process::port_t const& down_port = down.second;

      pipe->connect(up_name, up_port, down_name, down_port);
    }
  }

  return pipe;
}

cluster_info
::cluster_info(process::type_t const& type_,
               process_registry::description_t const& description_,
               process_ctor_t const& ctor_)
  : type(type_)
  , description(description_)
  , ctor(ctor_)
{
}

cluster_info
::~cluster_info()
{
}

cluster_info_t
bake_cluster_from_file(path_t const& fname)
{
  return bake_cluster_blocks(load_cluster_blocks_from_file(fname));
}

cluster_info_t
bake_cluster(std::istream& istr, path_t const& inc_root)
{
  return bake_cluster_blocks(load_cluster_blocks(istr, inc_root));
}

class cluster_bakery
  : public bakery_base
{
  public:
    cluster_bakery();
    ~cluster_bakery();

    using bakery_base::operator ();
    void operator () (cluster_pipe_block const& cluster_block_);

    class cluster_component_info_t
    {
      public:
        cluster_component_info_t();
        ~cluster_component_info_t();

        typedef std::vector<cluster_config_t> config_maps_t;
        typedef std::vector<cluster_input_t> input_maps_t;
        typedef std::vector<cluster_output_t> output_maps_t;

        config_maps_t m_configs;
        input_maps_t m_inputs;
        output_maps_t m_outputs;
    };
    typedef boost::optional<cluster_component_info_t> opt_cluster_component_info_t;

    process::type_t m_type;
    process_registry::description_t m_description;
    opt_cluster_component_info_t m_cluster;
};

static void dereference_static_providers(bakery_base::config_decls_t& bakery);

class cluster_creator
{
  public:
    cluster_creator(cluster_bakery const& bakery);
    ~cluster_creator();

    process_t operator () (kwiver::vital::config_block_sptr const& config) const;

    cluster_bakery const m_bakery;
  private:
    kwiver::vital::config_block_sptr m_default_config;
};

cluster_info_t
bake_cluster_blocks(cluster_blocks const& blocks)
{
  cluster_bakery bakery;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(bakery));

  if (bakery.m_processes.empty())
  {
    throw cluster_without_processes_exception();
  }

  cluster_bakery::opt_cluster_component_info_t const& opt_cluster = bakery.m_cluster;

  if (!opt_cluster)
  {
    throw missing_cluster_block_exception();
  }

  cluster_bakery::cluster_component_info_t const& cluster = *opt_cluster;

  if (cluster.m_inputs.empty() &&
      cluster.m_outputs.empty())
  {
    throw cluster_without_ports_exception();
  }

  bakery_base::config_decls_t& configs = bakery.m_configs;

  dereference_static_providers(configs);

  process::type_t const& type = bakery.m_type;
  process_registry::description_t const& description = bakery.m_description;
  process_ctor_t const ctor = cluster_creator(bakery);

  cluster_info_t const info = boost::make_shared<cluster_info>(type, description, ctor);

  return info;
}

kwiver::vital::config_block_sptr
extract_configuration(pipe_blocks const& blocks)
{
  pipe_bakery bakery;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(bakery));

  bakery_base::config_decls_t& configs = bakery.m_configs;

  return extract_configuration_from_decls(configs);
}

class provider_dereferencer
  : public boost::static_visitor<bakery_base::config_reference_t>
{
  public:
    provider_dereferencer();
    provider_dereferencer(kwiver::vital::config_block_sptr const conf);
    ~provider_dereferencer();

    bakery_base::config_reference_t operator () (kwiver::vital::config_block_value_t const& value) const;
    bakery_base::config_reference_t operator () (bakery_base::provider_request_t const& request) const;
  private:
    typedef std::map<config_provider_t, provider_t> provider_map_t;
    provider_map_t m_providers;
};

class ensure_provided
  : public boost::static_visitor<kwiver::vital::config_block_value_t>
{
  public:
    ensure_provided();
    ~ensure_provided();

    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& value) const;
    kwiver::vital::config_block_value_t operator () (bakery_base::provider_request_t const& request) const;
};

static void set_config_value(kwiver::vital::config_block_sptr conf, bakery_base::config_info_t const& flags, kwiver::vital::config_block_key_t const& key, kwiver::vital::config_block_value_t const& value);

class config_provider_sorter
  : public boost::static_visitor<>
{
  public:
    config_provider_sorter();
    ~config_provider_sorter();

    void operator () (kwiver::vital::config_block_key_t const& key, kwiver::vital::config_block_value_t const& value) const;
    void operator () (kwiver::vital::config_block_key_t const& key, bakery_base::provider_request_t const& request);

    kwiver::vital::config_block_keys_t sorted() const;
  private:
    class node_t
    {
      public:
        node_t();
        ~node_t();

        bool deref;
        kwiver::vital::config_block_key_t name;
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, node_t> config_graph_t;
    typedef boost::graph_traits<config_graph_t>::vertex_descriptor vertex_t;
    typedef std::vector<vertex_t> vertices_t;
    typedef std::map<kwiver::vital::config_block_key_t, vertex_t> vertex_map_t;
    typedef vertex_map_t::value_type vertex_entry_t;

    vertex_map_t m_vertex_map;
    config_graph_t m_graph;
};

kwiver::vital::config_block_sptr
extract_configuration_from_decls(bakery_base::config_decls_t& configs)
{
  dereference_static_providers(configs);

  kwiver::vital::config_block_sptr tmp_conf = kwiver::vital::config_block::empty_config();

  ensure_provided const ensure;

  {
    typedef std::set<kwiver::vital::config_block_key_t> unprovided_keys_t;

    unprovided_keys_t unprovided_keys;

    BOOST_FOREACH (bakery_base::config_decl_t& decl, configs)
    {
      kwiver::vital::config_block_key_t const& key = decl.first;

      if (unprovided_keys.count(key))
      {
        continue;
      }

      bakery_base::config_info_t const& info = decl.second;
      bakery_base::config_reference_t const& ref = info.reference;

      kwiver::vital::config_block_value_t val;

      // Only add provided configurations to the configuration.
      try
      {
        val = boost::apply_visitor(ensure, ref);
      }
      catch (unrecognized_provider_exception const /*e*/)
      {
        unprovided_keys.insert(key);

        continue;
      }

      set_config_value(tmp_conf, info, key, val);
    }
  }

  // Dereference configuration providers.
  {
    config_provider_sorter sorter;

    BOOST_FOREACH (bakery_base::config_decl_t& decl, configs)
    {
      kwiver::vital::config_block_key_t const& key = decl.first;
      bakery_base::config_info_t const& info = decl.second;
      bakery_base::config_reference_t const& ref = info.reference;

      /// \bug Why must this be done?
      typedef boost::variant<kwiver::vital::config_block_key_t> dummy_variant;

      dummy_variant const var = key;

      boost::apply_visitor(sorter, var, ref);
    }

    kwiver::vital::config_block_keys_t const keys = sorter.sorted();

    provider_dereferencer const deref(tmp_conf);

    /// \todo This is algorithmically naive, but I'm not sure if there's a better way.
    BOOST_FOREACH (kwiver::vital::config_block_key_t const& key, keys)
    {
      BOOST_FOREACH (bakery_base::config_decl_t& decl, configs)
      {
        kwiver::vital::config_block_key_t const& cur_key = decl.first;

        if (key != cur_key)
        {
          continue;
        }

        bakery_base::config_info_t& info = decl.second;
        bakery_base::config_reference_t& ref = info.reference;

        ref = boost::apply_visitor(deref, ref);

        kwiver::vital::config_block_value_t const val = boost::apply_visitor(ensure, ref);

        set_config_value(tmp_conf, info, key, val);
      }
    }
  }

  kwiver::vital::config_block_sptr conf = kwiver::vital::config_block::empty_config();

  BOOST_FOREACH (bakery_base::config_decl_t& decl, configs)
  {
    kwiver::vital::config_block_key_t const& key = decl.first;
    bakery_base::config_info_t const& info = decl.second;
    bakery_base::config_reference_t const& ref = info.reference;

    kwiver::vital::config_block_value_t val;

    try
    {
      val = boost::apply_visitor(ensure, ref);
    }
    catch (unrecognized_provider_exception const& e)
    {
      throw unrecognized_provider_exception(key, e.m_provider, e.m_index);
    }

    set_config_value(conf, info, key, val);
  }

  return conf;
}

bakery_base::config_info_t
::config_info_t(config_reference_t const& ref,
                bool ro,
                append_t app)
  : reference(ref)
  , read_only(ro)
  , append(app)
{
}

bakery_base::config_info_t
::~config_info_t()
{
}

bakery_base
::bakery_base()
  : m_configs()
  , m_processes()
  , m_connections()
{
}

bakery_base
::~bakery_base()
{
}

static kwiver::vital::config_block_key_t flatten_keys(kwiver::vital::config_block_keys_t const& keys);

void
bakery_base
::operator () (config_pipe_block const& config_block)
{
  kwiver::vital::config_block_key_t const root_key = flatten_keys(config_block.key);

  config_values_t const& values = config_block.values;

  BOOST_FOREACH (config_value_t const& value, values)
  {
    register_config_value(root_key, value);
  }
}

void
bakery_base
::operator () (process_pipe_block const& process_block)
{
  config_values_t const& values = process_block.config_values;

  BOOST_FOREACH (config_value_t const& value, values)
  {
    register_config_value(process_block.name, value);
  }

  m_processes.push_back(process_decl_t(process_block.name, process_block.type));
}

void
bakery_base
::operator () (connect_pipe_block const& connect_block)
{
  m_connections.push_back(process::connection_t(connect_block.from, connect_block.to));
}

void
bakery_base
::register_config_value(kwiver::vital::config_block_key_t const& root_key, config_value_t const& value)
{
  config_key_t const key = value.key;

  kwiver::vital::config_block_key_t const subkey = flatten_keys(key.key_path);

  config_reference_t c_value;

  if (key.options.provider)
  {
    c_value = provider_request_t(*key.options.provider, value.value);
  }
  else
  {
    c_value = value.value;
  }

  kwiver::vital::config_block_key_t const full_key = root_key + kwiver::vital::config_block::block_sep + subkey;

  bool is_readonly = false;
  config_info_t::append_t append = config_info_t::append_none;

#define APPEND_CHECK(flag)                                             \
  do                                                                   \
  {                                                                    \
    if (append != config_info_t::append_none)                          \
    {                                                                  \
      std::string const reason = "The \'" + flag + "\' flag cannot "   \
                                 "be used with other appending flags"; \
                                                                       \
      throw config_flag_mismatch_exception(full_key, reason);          \
    }                                                                  \
  } while (false)

  if (key.options.flags)
  {
    BOOST_FOREACH (config_flag_t const& flag, *key.options.flags)
    {
      if (flag == flag_read_only)
      {
        is_readonly = true;
      }
      else if (flag == flag_append)
      {
        APPEND_CHECK(flag_append);

        append = config_info_t::append_string;
      }
      else if (boost::starts_with(flag, flag_append_prefix))
      {
        APPEND_CHECK(flag);

        config_flag_t const& kind = flag.substr(flag_append_prefix.size());

        if (kind == flag_append_comma)
        {
          append = config_info_t::append_comma;
        }
        else if (kind == flag_append_space)
        {
          append = config_info_t::append_space;
        }
        else if (kind == flag_append_path)
        {
          append = config_info_t::append_path;
        }
        else
        {
          throw unrecognized_config_flag_exception(full_key, flag);
        }
      }
      else if (flag == flag_tunable)
      {
        // Ignore here (but don't error).
      }
      else
      {
        throw unrecognized_config_flag_exception(full_key, flag);
      }
    }
  }

#undef APPEND_CHECK

  config_info_t const info = config_info_t(c_value, is_readonly, append);

  config_decl_t const decl = config_decl_t(full_key, info);

  m_configs.push_back(decl);
}

pipe_bakery
::pipe_bakery()
  : bakery_base()
{
}

pipe_bakery
::~pipe_bakery()
{
}

cluster_bakery
::cluster_bakery()
  : bakery_base()
{
}

cluster_bakery
::~cluster_bakery()
{
}

class cluster_splitter
  : public boost::static_visitor<>
{
  public:
    cluster_splitter(cluster_bakery::cluster_component_info_t& info);
    ~cluster_splitter();

    void operator () (cluster_config_t const& config_block);
    void operator () (cluster_input_t const& input_block);
    void operator () (cluster_output_t const& output_block);

    cluster_bakery::cluster_component_info_t& m_info;
  private:
    typedef std::set<process::port_t> unique_ports_t;

    unique_ports_t m_input_ports;
    unique_ports_t m_output_ports;
};

void
cluster_bakery
::operator () (cluster_pipe_block const& cluster_block_)
{
  if (m_cluster)
  {
    throw multiple_cluster_blocks_exception();
  }

  m_type = cluster_block_.type;
  m_description = cluster_block_.description;

  cluster_component_info_t cluster;

  cluster_subblocks_t const& subblocks = cluster_block_.subblocks;

  cluster_splitter splitter(cluster);

  std::for_each(subblocks.begin(), subblocks.end(), boost::apply_visitor(splitter));

  m_cluster = cluster;

  cluster_component_info_t::config_maps_t const& values = cluster.m_configs;
  BOOST_FOREACH (cluster_config_t const& value, values)
  {
    config_value_t const& config_value = value.config_value;

    register_config_value(m_type, config_value);
  }
}

cluster_bakery::cluster_component_info_t
::cluster_component_info_t()
{
}

cluster_bakery::cluster_component_info_t
::~cluster_component_info_t()
{
}

void
dereference_static_providers(bakery_base::config_decls_t& configs)
{
  provider_dereferencer const deref;

  BOOST_FOREACH (bakery_base::config_decl_t& decl, configs)
  {
    bakery_base::config_info_t& info = decl.second;
    bakery_base::config_reference_t& ref = info.reference;

    ref = boost::apply_visitor(deref, ref);
  }
}

cluster_creator
::cluster_creator(cluster_bakery const& bakery)
  : m_bakery(bakery)
{
  bakery_base::config_decls_t default_configs = m_bakery.m_configs;

  m_default_config = extract_configuration_from_decls(default_configs);
}

cluster_creator
::~cluster_creator()
{
}

class loaded_cluster
  : public process_cluster
{
  public:
    loaded_cluster(kwiver::vital::config_block_sptr const& config);
    ~loaded_cluster();

    friend class cluster_creator;
};

class provided_by_cluster
{
  public:
    provided_by_cluster(process::type_t const& name, process::names_t const& procs);
    ~provided_by_cluster();

    bool operator () (bakery_base::config_decl_t const& decl) const;
  private:
    process::type_t const m_name;
    process::names_t const m_procs;
};

class extract_literal_value
  : public boost::static_visitor<kwiver::vital::config_block_value_t>
{
  public:
    extract_literal_value();
    ~extract_literal_value();

    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& value) const;
    kwiver::vital::config_block_value_t operator () (bakery_base::provider_request_t const& request) const;
};

#if __cplusplus >= 201103L
#define COPY_IF std::copy_if
// XXX(boost): 1.52.0
#elif BOOST_VERSION >= 105200
#define COPY_IF boost::algorithm::copy_if
#else
#define COPY_IF sprokit::copy_if

#define USE_CUSTOM_COPY_IF

template <typename InputIterator, typename OutputIterator, typename UnaryPredicate>
static OutputIterator copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred);
#endif

process_t
cluster_creator
::operator () (kwiver::vital::config_block_sptr const& config) const
{
  bakery_base::config_decls_t all_configs = m_bakery.m_configs;

  process::type_t const& type = m_bakery.m_type;

  process::names_t proc_names;

  BOOST_FOREACH (bakery_base::process_decl_t const& proc_decl, m_bakery.m_processes)
  {
    process::name_t const& proc_name = proc_decl.first;

    proc_names.push_back(proc_name);
  }

  provided_by_cluster const mapping_filter(type, proc_names);

  bakery_base::config_decls_t mapped_decls;

  // Copy out configuration settings which are mapped by the cluster.
  COPY_IF(all_configs.begin(), all_configs.end(), std::back_inserter(mapped_decls), mapping_filter);

  // Append the given configuration to the declarations from the parsed blocks.
  kwiver::vital::config_block_keys_t const& keys = config->available_values();
  BOOST_FOREACH (kwiver::vital::config_block_key_t const& key, keys)
  {
    kwiver::vital::config_block_value_t const value = config->get_value<kwiver::vital::config_block_value_t>(key);
    bakery_base::config_reference_t const ref = bakery_base::config_reference_t(value);
    bool const is_read_only = config->is_read_only(key);
    bakery_base::config_info_t const info = bakery_base::config_info_t(ref, is_read_only, bakery_base::config_info_t::append_none);
    kwiver::vital::config_block_key_t const full_key = kwiver::vital::config_block_key_t(type) +
      kwiver::vital::config_block::block_sep + key;
    bakery_base::config_decl_t const decl = bakery_base::config_decl_t(full_key, info);

    all_configs.push_back(decl);
  }

  kwiver::vital::config_block_sptr const full_config = extract_configuration_from_decls(all_configs);

  typedef boost::shared_ptr<loaded_cluster> loaded_cluster_t;

  // Pull out the main config block to the top-level.
  kwiver::vital::config_block_sptr const cluster_config = full_config->subblock_view(type);
  full_config->merge_config(cluster_config);

  loaded_cluster_t const cluster = boost::make_shared<loaded_cluster>(full_config);

  cluster_bakery::opt_cluster_component_info_t const& opt_info = m_bakery.m_cluster;

  if (!opt_info)
  {
    static std::string const reason = "Failed to catch missing cluster block earlier";

    throw std::logic_error(reason);
  }

  cluster_bakery::cluster_component_info_t const& info = *opt_info;

  kwiver::vital::config_block_sptr const main_config = m_default_config->subblock_view(type);

  // Declare configuration values.
  BOOST_FOREACH (cluster_config_t const& conf, info.m_configs)
  {
    config_value_t const& config_value = conf.config_value;
    config_key_t const& config_key = config_value.key;
    kwiver::vital::config_block_keys_t const& key_path = config_key.key_path;
    kwiver::vital::config_block_key_t const& key = flatten_keys(key_path);
    kwiver::vital::config_block_value_t const& value = main_config->get_value<kwiver::vital::config_block_value_t>(key);
    kwiver::vital::config_block_description_t const& description = conf.description;
    config_key_options_t const& options = config_key.options;
    bool tunable = false;

    if (options.flags)
    {
      config_flags_t const& flags = *options.flags;

      tunable = (0 != std::count(flags.begin(), flags.end(), flag_tunable));
    }

    cluster->declare_configuration_key(
      key,
      value,
      description,
      tunable);
  }

  extract_literal_value const literal_value = extract_literal_value();

  // Add config mappings.
  BOOST_FOREACH (bakery_base::config_decl_t const& decl, mapped_decls)
  {
    kwiver::vital::config_block_key_t const& key = decl.first;
    bakery_base::config_info_t const& mapping_info = decl.second;
    bakery_base::config_reference_t const& ref = mapping_info.reference;

    kwiver::vital::config_block_value_t const value = boost::apply_visitor(literal_value, ref);

    kwiver::vital::config_block_keys_t mapped_key_path;
    kwiver::vital::config_block_keys_t source_key_path;

    /// \bug Does not work if (kwiver::vital::config_block::block_sep.size() != 1).
    boost::split(mapped_key_path, key, boost::is_any_of(kwiver::vital::config_block::block_sep));
    /// \bug Does not work if (kwiver::vital::config_block::block_sep.size() != 1).
    boost::split(source_key_path, value, boost::is_any_of(kwiver::vital::config_block::block_sep));

    if (mapped_key_path.size() < 2)
    {
      /// \todo Error.

      continue;
    }

    if (source_key_path.size() < 2)
    {
      /// \todo Error.

      continue;
    }

    kwiver::vital::config_block_key_t const mapped_name = mapped_key_path[0];
    mapped_key_path.erase(mapped_key_path.begin());

    kwiver::vital::config_block_key_t const mapped_key = flatten_keys(mapped_key_path);

    source_key_path.erase(source_key_path.begin());
    kwiver::vital::config_block_key_t const source_key = flatten_keys(source_key_path);

    cluster->map_config(source_key, mapped_name, mapped_key);
  }

  // Add processes.
  BOOST_FOREACH (bakery_base::process_decl_t const& proc_decl, m_bakery.m_processes)
  {
    process::name_t const& proc_name = proc_decl.first;
    process::type_t const& proc_type = proc_decl.second;

    kwiver::vital::config_block_sptr const proc_config = full_config->subblock_view(proc_name);

    cluster->add_process(proc_name, proc_type, proc_config);
  }

  // Add input ports.
  {
    process::port_flags_t const input_flags;

    BOOST_FOREACH (cluster_input_t const& input, info.m_inputs)
    {
      process::port_description_t const& description = input.description;
      process::port_t const& port = input.from;

      cluster->declare_input_port(
        port,
        /// \todo How to declare a port's type?
        process::type_any,
        input_flags,
        description);

      process::port_addrs_t const& addrs = input.targets;

      BOOST_FOREACH (process::port_addr_t const& addr, addrs)
      {
        process::name_t const& mapped_name = addr.first;
        process::port_t const& mapped_port = addr.second;

        cluster->map_input(
          port,
          mapped_name,
          mapped_port);
      }
    }
  }

  // Add output ports.
  {
    process::port_flags_t const output_flags;

    BOOST_FOREACH (cluster_output_t const& output, info.m_outputs)
    {
      process::port_description_t const& description = output.description;
      process::port_t const& port = output.to;

      cluster->declare_output_port(
        port,
        /// \todo How to declare a port's type?
        process::type_any,
        output_flags,
        description);

      process::port_addr_t const& addr = output.from;

      process::name_t const& mapped_name = addr.first;
      process::port_t const& mapped_port = addr.second;

      cluster->map_output(
        port,
        mapped_name,
        mapped_port);
    }
  }

  // Add connections.
  BOOST_FOREACH (process::connection_t const& connection, m_bakery.m_connections)
  {
    process::port_addr_t const& upstream_addr = connection.first;
    process::port_addr_t const& downstream_addr = connection.second;

    process::name_t const& upstream_name = upstream_addr.first;
    process::port_t const& upstream_port = upstream_addr.second;
    process::name_t const& downstream_name = downstream_addr.first;
    process::port_t const& downstream_port = downstream_addr.second;

    cluster->connect(upstream_name, upstream_port,
                     downstream_name, downstream_port);
  }

  return cluster;
}

provider_dereferencer
::provider_dereferencer()
  : m_providers()
{
  m_providers[provider_system] = boost::make_shared<system_provider>();
  m_providers[provider_environment] = boost::make_shared<environment_provider>();
}

provider_dereferencer
::provider_dereferencer(kwiver::vital::config_block_sptr const conf)
  : m_providers()
{
  m_providers[provider_config] = boost::make_shared<config_provider>(conf);
}

provider_dereferencer
::~provider_dereferencer()
{
}

bakery_base::config_reference_t
provider_dereferencer
::operator () (kwiver::vital::config_block_value_t const& value) const
{
  return value;
}

bakery_base::config_reference_t
provider_dereferencer
::operator () (bakery_base::provider_request_t const& request) const
{
  config_provider_t const& provider_name = request.first;
  provider_map_t::const_iterator const i = m_providers.find(provider_name);

  if (i == m_providers.end())
  {
    return request;
  }

  provider_t const& provider = i->second;
  kwiver::vital::config_block_value_t const& value = request.second;

  return (*provider)(value);
}

ensure_provided
::ensure_provided()
{
}

ensure_provided
::~ensure_provided()
{
}

kwiver::vital::config_block_value_t
ensure_provided
::operator () (kwiver::vital::config_block_value_t const& value) const
{
  return value;
}

kwiver::vital::config_block_value_t
ensure_provided
::operator () (bakery_base::provider_request_t const& request) const
{
  config_provider_t const& provider = request.first;
  kwiver::vital::config_block_value_t const& value = request.second;

  throw unrecognized_provider_exception("(unknown)", provider, value);
}

void
set_config_value(kwiver::vital::config_block_sptr conf, bakery_base::config_info_t const& flags, kwiver::vital::config_block_key_t const& key, kwiver::vital::config_block_value_t const& value)
{
  kwiver::vital::config_block_value_t val = value;

  kwiver::vital::config_block_value_t const cur_val = conf->get_value(key, kwiver::vital::config_block_value_t());
  bool const has_cur_val = !cur_val.empty();

  switch (flags.append)
  {
    case bakery_base::config_info_t::append_string:
      val = cur_val + val;
      break;
    case bakery_base::config_info_t::append_comma:
      if (has_cur_val)
      {
        val = cur_val + "," + val;
      }
      break;
    case bakery_base::config_info_t::append_space:
      if (has_cur_val)
      {
        val = cur_val + " " + val;
      }
      break;
    case bakery_base::config_info_t::append_path:
      {
        path_t const base_path = path_t(has_cur_val ? cur_val : ".");
        path_t const val_path = path_t(val);
        path_t const new_path = base_path / val_path;

        val = new_path.string<kwiver::vital::config_block_value_t>();
      }
      break;
    case bakery_base::config_info_t::append_none:
    default:
      break;
  }

  conf->set_value(key, val);

  if (flags.read_only)
  {
    conf->mark_read_only(key);
  }
}

config_provider_sorter
::config_provider_sorter()
  : m_vertex_map()
  , m_graph()
{
}

config_provider_sorter
::~config_provider_sorter()
{
}

kwiver::vital::config_block_keys_t
config_provider_sorter
::sorted() const
{
  vertices_t vertices;

  try
  {
    boost::topological_sort(m_graph, std::back_inserter(vertices));
  }
  catch (boost::not_a_dag const&)
  {
    throw circular_config_provide_exception();
  }

  kwiver::vital::config_block_keys_t keys;

  BOOST_FOREACH (vertex_t const& vertex, vertices)
  {
    node_t const& node = m_graph[vertex];

    if (node.deref)
    {
      keys.push_back(node.name);
    }
  }

  return keys;
}

void
config_provider_sorter
::operator () (kwiver::vital::config_block_key_t const& /*key*/, kwiver::vital::config_block_value_t const& /*value*/) const
{
}

void
config_provider_sorter
::operator () (kwiver::vital::config_block_key_t const& key, bakery_base::provider_request_t const& request)
{
  config_provider_t const& provider = request.first;
  kwiver::vital::config_block_value_t const& value = request.second;

  if (provider != provider_config)
  {
    return;
  }

  kwiver::vital::config_block_key_t const& target_key = kwiver::vital::config_block_key_t(value);

  typedef std::pair<vertex_map_t::iterator, bool> insertion_t;

  insertion_t from_iter = m_vertex_map.insert(vertex_entry_t(key, vertex_t()));
  insertion_t to_iter = m_vertex_map.insert(vertex_entry_t(target_key, vertex_t()));

  bool const& from_inserted = from_iter.second;
  bool const& to_inserted = to_iter.second;

  vertex_map_t::iterator& from = from_iter.first;
  vertex_map_t::iterator& to = to_iter.first;

  vertex_t& from_vertex = from->second;
  vertex_t& to_vertex = to->second;

  if (from_inserted)
  {
    from_vertex = boost::add_vertex(m_graph);
    m_graph[from_vertex].name = key;
  }

  if (to_inserted)
  {
    to_vertex = boost::add_vertex(m_graph);
    m_graph[to_vertex].name = target_key;
  }

  m_graph[from_vertex].deref = true;

  boost::add_edge(from_vertex, to_vertex, m_graph);
}

config_provider_sorter::node_t
::node_t()
  : deref(false)
  , name()
{
}

config_provider_sorter::node_t
::~node_t()
{
}

kwiver::vital::config_block_key_t
flatten_keys(kwiver::vital::config_block_keys_t const& keys)
{
  return boost::join(keys, kwiver::vital::config_block::block_sep);
}

cluster_splitter
::cluster_splitter(cluster_bakery::cluster_component_info_t& info)
  : m_info(info)
{
}

cluster_splitter
::~cluster_splitter()
{
}

void
cluster_splitter
::operator () (cluster_config_t const& config_block)
{
  m_info.m_configs.push_back(config_block);
}

void
cluster_splitter
::operator () (cluster_input_t const& input_block)
{
  process::port_t const& port = input_block.from;

  if (m_input_ports.count(port))
  {
    throw duplicate_cluster_input_port_exception(port);
  }

  m_input_ports.insert(port);

  m_info.m_inputs.push_back(input_block);
}

void
cluster_splitter
::operator () (cluster_output_t const& output_block)
{
  process::port_t const& port = output_block.to;

  if (m_output_ports.count(port))
  {
    throw duplicate_cluster_output_port_exception(port);
  }

  m_output_ports.insert(port);

  m_info.m_outputs.push_back(output_block);
}

loaded_cluster
::loaded_cluster(kwiver::vital::config_block_sptr const& config)
  : process_cluster(config)
{
}

loaded_cluster
::~loaded_cluster()
{
}

provided_by_cluster
::provided_by_cluster(process::type_t const& name, process::names_t const& procs)
  : m_name(name)
  , m_procs(procs)
{
}

provided_by_cluster
::~provided_by_cluster()
{
}

class check_provider
  : public boost::static_visitor<bool>
{
  public:
    check_provider(config_provider_t const& provider);
    ~check_provider();

    bool operator () (kwiver::vital::config_block_value_t const& value) const;
    bool operator () (bakery_base::provider_request_t const& request) const;
  private:
    config_provider_t const m_provider;
};

bool
provided_by_cluster
::operator () (bakery_base::config_decl_t const& decl) const
{
  bakery_base::config_info_t const& info = decl.second;

  // Mapped configurations must be read-only.
  if (!info.read_only)
  {
    return false;
  }

  // Mapped configurations must be a provider_config request.
  check_provider const check = check_provider(provider_config);

  bakery_base::config_reference_t const& ref = info.reference;
  bool const conf_provided = boost::apply_visitor(check, ref);

  if (!conf_provided)
  {
    return false;
  }

  extract_literal_value const literal_value = extract_literal_value();

  kwiver::vital::config_block_value_t const value = boost::apply_visitor(literal_value, ref);

  // It must be mapped to the the actual cluster.
  if (!boost::starts_with(value, m_name + kwiver::vital::config_block::block_sep))
  {
    return false;
  }

  /**
   * \todo There should be at least a warning that if the target is being
   * provided by a tunable parameter on the cluster that this will likely not
   * work as intended.
   */

  kwiver::vital::config_block_key_t const& key = decl.first;

  kwiver::vital::config_block_keys_t key_path;

  /// \bug Does not work if (kwiver::vital::config_block::block_sep.size() != 1).
  boost::split(key_path, key, boost::is_any_of(kwiver::vital::config_block::block_sep));

  bool const is_proc = (0 != std::count(m_procs.begin(), m_procs.end(), key_path[0]));

  if (!is_proc)
  {
    // We can't map to non-processes.
    return false;
  }

  return true;
}

extract_literal_value
::extract_literal_value()
{
}

extract_literal_value
::~extract_literal_value()
{
}

kwiver::vital::config_block_value_t
extract_literal_value
::operator () (kwiver::vital::config_block_value_t const& value) const
{
  return value;
}

kwiver::vital::config_block_value_t
extract_literal_value
::operator () (bakery_base::provider_request_t const& request) const
{
  kwiver::vital::config_block_value_t const& value = request.second;

  return value;
}

#ifdef USE_CUSTOM_COPY_IF
template <typename InputIterator, typename OutputIterator, typename UnaryPredicate>
OutputIterator
copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred)
{
  while (first != last)
  {
    if (pred(*first))
    {
      *result = *first;
      ++result;
    }

    ++first;
  }

  return result;
}
#endif

check_provider
::check_provider(config_provider_t const& provider)
  : m_provider(provider)
{
}

check_provider
::~check_provider()
{
}

bool
check_provider
::operator () (kwiver::vital::config_block_value_t const& /*value*/) const
{
  return false;
}

bool
check_provider
::operator () (bakery_base::provider_request_t const& request) const
{
  config_provider_t const& provider = request.first;

  return (m_provider == provider);
}

}
