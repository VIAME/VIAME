/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#include <boost/variant.hpp>

#include <fstream>
#include <iostream>

#include <cstdlib>

class config_printer
  : public boost::static_visitor<>
{
  public:
    config_printer(std::ostream& ostr, vistk::pipeline_t const& pipe, vistk::config_t const& conf);
    ~config_printer();

    void operator () (vistk::config_pipe_block const& config_block) const;
    void operator () (vistk::process_pipe_block const& process_block) const;
    void operator () (vistk::connect_pipe_block const& connect_block) const;
    void operator () (vistk::group_pipe_block const& group_block) const;
  private:
    void print_config_value(vistk::config_value_t const& config_value) const;

    std::ostream& m_ostr;
    vistk::pipeline_t const m_pipe;
    vistk::config_t const m_config;
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

  std::ostream* postr;
  std::ofstream fout;

  vistk::path_t const opath = vm["output"].as<vistk::path_t>();

  if (opath.native() == vistk::path_t("-"))
  {
    postr = &std::cout;
  }
  else
  {
    fout.open(opath.native().c_str());

    if (fout.bad())
    {
      std::cerr << "Error: Unable to open output file" << std::endl;

      return EXIT_FAILURE;
    }

    postr = &fout;
  }

  std::ostream& ostr = *postr;

  config_printer const printer(ostr, pipe, config);

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(printer));

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

void
config_printer
::operator () (vistk::config_pipe_block const& config_block) const
{
  vistk::config::keys_t const& keys = config_block.key;
  vistk::config_values_t const& values = config_block.values;

  vistk::config::key_t const key_path = boost::join(keys, vistk::config::block_sep);

  m_ostr << "config " << key_path << std::endl;

  key_printer const printer(m_ostr);

  std::for_each(values.begin(), values.end(), printer);

  m_ostr << std::endl;
}

void
config_printer
::operator () (vistk::process_pipe_block const& process_block) const
{
  vistk::process::name_t const& name = process_block.name;
  vistk::process::type_t const& type = process_block.type;
  vistk::config_values_t const& values = process_block.config_values;

  m_ostr << "process " << name << std::endl;
  m_ostr << "  :: " << type << std::endl;

  key_printer const printer(m_ostr);

  std::for_each(values.begin(), values.end(), printer);

  vistk::process_t const proc = m_pipe->process_by_name(name);

  vistk::config::keys_t const keys = proc->available_config();

  BOOST_FOREACH (vistk::config::key_t const& key, keys)
  {
    if (boost::starts_with(key, "_"))
    {
      continue;
    }

    m_ostr << std::endl;

    vistk::process::conf_info_t const& info = proc->config_info(key);

    vistk::config::description_t const desc = boost::replace_all_copy(info->description, "\n", "\n  #   ");

    m_ostr << "  # Key: " << key << std::endl;

    m_ostr << "  # Description: " << desc << std::endl;

    vistk::config::value_t const& def = info->def;

    if (def.empty())
    {
      m_ostr << "  # No default value" << std::endl;
    }
    else
    {
      m_ostr << "  # Default value: " << def << std::endl;
    }

    vistk::config::key_t const resolved_key = name + vistk::config::block_sep + key;

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

class group_printer
: public boost::static_visitor<>
{
  public:
    group_printer(std::ostream& ostr);
    ~group_printer();

    void operator () (vistk::config_value_t const& config_value) const;
    void operator () (vistk::input_map_t const& input_map) const;
    void operator () (vistk::output_map_t const& output_map) const;
  private:
    std::ostream& m_ostr;
};

void
config_printer
::operator () (vistk::group_pipe_block const& group_block) const
{
  vistk::process::name_t const& name = group_block.name;
  vistk::group_subblocks_t const& subblocks = group_block.subblocks;

  m_ostr << "group " << name << std::endl;

  group_printer const printer(m_ostr);

  std::for_each(subblocks.begin(), subblocks.end(), boost::apply_visitor(printer));

  m_ostr << std::endl;
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

group_printer
::group_printer(std::ostream& ostr)
  : m_ostr(ostr)
{
}

group_printer
::~group_printer()
{
}

void
group_printer
::operator () (vistk::config_value_t const& config_value) const
{
  key_printer const printer(m_ostr);

  printer(config_value);
}

void
group_printer
::operator () (vistk::input_map_t const& input_map) const
{
  vistk::map_options_t const& options = input_map.options;
  vistk::process::port_t const& from = input_map.from;
  vistk::process::port_addr_t const& to = input_map.to;

  boost::optional<vistk::process::port_flags_t> const& flags = options.flags;

  vistk::process::name_t const& to_name = to.first;
  vistk::process::port_t const& to_port = to.second;

  m_ostr << "  imap";

  if (flags)
  {
    vistk::config_flag_t const flag_list = boost::join(*flags, ",");

    m_ostr << "[" << flag_list << "]";
  }

  m_ostr << " from " << from << std::endl;
  m_ostr << "  to " << to_name << "." << to_port << std::endl;
}

void
group_printer
::operator () (vistk::output_map_t const& output_map) const
{
  vistk::map_options_t const& options = output_map.options;
  vistk::process::port_addr_t const& from = output_map.from;
  vistk::process::port_t const& to = output_map.to;

  boost::optional<vistk::process::port_flags_t> const& flags = options.flags;

  vistk::process::name_t const& from_name = from.first;
  vistk::process::port_t const& from_port = from.second;

  m_ostr << "  omap";

  if (flags)
  {
    vistk::config_flag_t const flag_list = boost::join(*flags, ",");

    m_ostr << "[" << flag_list << "]";
  }

  m_ostr << " from " << from_name << "." << from_port << std::endl;
  m_ostr << "  to " << to << std::endl;
}
