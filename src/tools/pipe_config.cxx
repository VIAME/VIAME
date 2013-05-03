/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_io.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_cluster.h>
#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>

#include <cstdlib>

class config_printer
  : public boost::static_visitor<>
{
  public:
    config_printer(std::ostream& ostr, vistk::pipeline_t const& pipe, vistk::config_t const& conf);
    ~config_printer();

    void operator () (vistk::config_pipe_block const& config_block);
    void operator () (vistk::process_pipe_block const& process_block);
    void operator () (vistk::connect_pipe_block const& connect_block) const;

    void output_process_defaults();
  private:
    void print_config_value(vistk::config_value_t const& config_value) const;
    void output_process_by_name(vistk::process::name_t const& name, bool fatal_if_no_process);
    void output_process_block(vistk::process_t const& name, std::string const& kind);
    void output_process(vistk::process_t const& proc);

    typedef std::set<vistk::process::name_t> process_set_t;

    std::ostream& m_ostr;
    vistk::pipeline_t const m_pipe;
    vistk::config_t const m_config;
    process_set_t m_visited;
};

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(pipeline_common_options())
    .add(pipeline_input_options())
    .add(pipeline_output_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  pipeline_builder const builder(vm, desc);

  vistk::pipeline_t const pipe = builder.pipeline();
  vistk::config_t const config = builder.config();
  vistk::pipe_blocks const blocks = builder.blocks();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  vistk::path_t const opath = vm["output"].as<vistk::path_t>();

  ostream_t const ostr = open_ostream(opath);

  config_printer printer(*ostr, pipe, config);

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(printer));

  printer.output_process_defaults();

  return EXIT_SUCCESS;
}

config_printer
::config_printer(std::ostream& ostr, vistk::pipeline_t const& pipe, vistk::config_t const& conf)
  : m_ostr(ostr)
  , m_pipe(pipe)
  , m_config(conf)
{
}

config_printer
::~config_printer()
{
}

class key_printer
{
  public:
    key_printer(std::ostream& ostr);
    ~key_printer();

    void operator () (vistk::config_value_t const& config_value) const;
  private:
    std::ostream& m_ostr;
};

static vistk::process::name_t denormalize_name(vistk::process::name_t const& name);

void
config_printer
::operator () (vistk::config_pipe_block const& config_block)
{
  vistk::config::keys_t const& keys = config_block.key;
  vistk::config_values_t const& values = config_block.values;

  vistk::config::key_t const key_path = boost::join(keys, vistk::config::block_sep);

  m_ostr << "config " << key_path << std::endl;

  key_printer const printer(m_ostr);

  std::for_each(values.begin(), values.end(), printer);

  vistk::process::name_t const denorm_name = denormalize_name(key_path);

  output_process_by_name(denorm_name, false);
}

static vistk::process::name_t normalize_name(vistk::process::name_t const& name);

void
config_printer
::operator () (vistk::process_pipe_block const& process_block)
{
  vistk::process::name_t const& name = process_block.name;
  vistk::process::type_t const& type = process_block.type;
  vistk::config_values_t const& values = process_block.config_values;

  m_ostr << "process " << name << std::endl;
  m_ostr << "  :: " << type << std::endl;

  key_printer const printer(m_ostr);

  std::for_each(values.begin(), values.end(), printer);

  output_process_by_name(name, true);
}

void
config_printer
::operator () (vistk::connect_pipe_block const& connect_block) const
{
  vistk::process::port_addr_t const& upstream_addr = connect_block.from;
  vistk::process::port_addr_t const& downstream_addr = connect_block.to;

  vistk::process::name_t const& upstream_name = upstream_addr.first;
  vistk::process::port_t const& upstream_port = upstream_addr.second;
  vistk::process::name_t const& downstream_name = downstream_addr.first;
  vistk::process::port_t const& downstream_port = downstream_addr.second;

  m_ostr << "connect from " << upstream_name << "." << upstream_port << std::endl;
  m_ostr << "        to   " << downstream_name << "." << downstream_port << std::endl;

  m_ostr << std::endl;
}

void
config_printer
::output_process_defaults()
{
  vistk::process::names_t const cluster_names = m_pipe->cluster_names();

  BOOST_FOREACH (vistk::process::name_t const& name, cluster_names)
  {
    if (m_visited.count(name))
    {
      continue;
    }

    vistk::process_cluster_t const cluster = m_pipe->cluster_by_name(name);
    vistk::process_t const proc = boost::static_pointer_cast<vistk::process>(cluster);

    static std::string const kind = "cluster";

    output_process_block(proc, kind);
  }

  vistk::process::names_t const process_names = m_pipe->process_names();

  BOOST_FOREACH (vistk::process::name_t const& name, process_names)
  {
    if (m_visited.count(name))
    {
      continue;
    }

    vistk::process_t const proc = m_pipe->process_by_name(name);

    static std::string const kind = "process";

    output_process_block(proc, kind);
  }
}

void
config_printer
::output_process_by_name(vistk::process::name_t const& name, bool fatal_if_no_process)
{
  vistk::process_t proc;

  if (!m_visited.count(name))
  {
    try
    {
      proc = m_pipe->process_by_name(name);
    }
    catch (vistk::no_such_process_exception const& /*e*/)
    {
      try
      {
        vistk::process_cluster_t const cluster = m_pipe->cluster_by_name(name);

        proc = boost::static_pointer_cast<vistk::process>(cluster);
      }
      catch (vistk::no_such_process_exception const& /*e*/)
      {
        if (fatal_if_no_process)
        {
          std::string const reason = "A process block did not result in a process being "
                                     "added to the pipeline: " + name;

          throw std::logic_error(reason);
        }
      }
    }
  }

  if (proc)
  {
    output_process(proc);
  }
  else
  {
    m_ostr << std::endl;
  }
}

void
config_printer
::output_process_block(vistk::process_t const& proc, std::string const& kind)
{
  vistk::process::name_t const name = proc->name();
  vistk::process::type_t const type = proc->type();
  vistk::process::name_t const norm_name = normalize_name(name);

  m_ostr << "# Defaults for \'" << name << "\' " << kind << ":" << std::endl;
  m_ostr << "config " << norm_name << std::endl;
  m_ostr << "#  :: " << type << std::endl;

  output_process(proc);
}

void
config_printer
::output_process(vistk::process_t const& proc)
{
  vistk::process::name_t const name = proc->name();
  vistk::config::keys_t const keys = proc->available_config();
  vistk::config::keys_t const tunable_keys = proc->available_tunable_config();
  vistk::process::name_t const norm_name = normalize_name(name);

  BOOST_FOREACH (vistk::config::key_t const& key, keys)
  {
    if (boost::starts_with(key, "_"))
    {
      continue;
    }

    m_ostr << std::endl;

    vistk::process::conf_info_t const& info = proc->config_info(key);

    vistk::config::description_t const desc = boost::replace_all_copy(info->description, "\n", "\n  #   ");

    bool const is_tunable = std::count(tunable_keys.begin(), tunable_keys.end(), key);
    std::string const tunable = (is_tunable ? "yes" : "no");

    m_ostr << "  # Key: " << key << std::endl;
    m_ostr << "  # Description: " << desc << std::endl;
    m_ostr << "  # Tunable: " << tunable << std::endl;

    vistk::config::value_t const& def = info->def;

    if (def.empty())
    {
      m_ostr << "  # No default value" << std::endl;
    }
    else
    {
      m_ostr << "  # Default value: " << def << std::endl;
    }

    vistk::config::key_t const resolved_key = norm_name + vistk::config::block_sep + key;

    if (m_config->has_value(resolved_key))
    {
      vistk::config::value_t const cur_value = m_config->get_value<vistk::config::value_t>(resolved_key);

      m_ostr << "  # Current value: " << cur_value << std::endl;
    }
    else
    {
      m_ostr << "  # No current value" << std::endl;
    }
  }

  m_ostr << std::endl;

  m_visited.insert(name);
}

key_printer
::key_printer(std::ostream& ostr)
  : m_ostr(ostr)
{
}

key_printer
::~key_printer()
{
}

void
key_printer
::operator () (vistk::config_value_t const& config_value) const
{
  vistk::config_key_t const& key = config_value.key;
  vistk::config::value_t const& value = config_value.value;

  vistk::config::keys_t const& keys = key.key_path;
  vistk::config_key_options_t const& options = key.options;

  vistk::config::key_t const key_path = boost::join(keys, vistk::config::block_sep);

  boost::optional<vistk::config_flags_t> const& flags = options.flags;
  boost::optional<vistk::config_provider_t> const& provider = options.provider;

  m_ostr << "  " << vistk::config::block_sep << key_path;

  if (flags)
  {
    vistk::config_flag_t const flag_list = boost::join(*flags, ",");

    m_ostr << "[" << flag_list << "]";
  }

  if (provider)
  {
    m_ostr << "{" << *provider << "}";
  }

  m_ostr << " " << value << std::endl;
}

vistk::process::name_t
denormalize_name(vistk::process::name_t const& name)
{
  vistk::process::name_t denorm_name;

  std::replace_copy_if(name.begin(), name.end(),
                       std::back_inserter(denorm_name),
                       boost::is_any_of(":"), '/');

  return denorm_name;
}

vistk::process::name_t
normalize_name(vistk::process::name_t const& name)
{
  vistk::process::name_t norm_name;

  std::replace_copy_if(name.begin(), name.end(),
                       std::back_inserter(norm_name),
                       boost::is_any_of("/"), ':');

  return norm_name;
}
