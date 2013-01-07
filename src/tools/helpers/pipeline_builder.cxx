/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_builder.h"

#include "tool_usage.h"

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler_registry.h>

#include <vistk/utilities/path.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/bind.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

static std::string const split_str = "=";

}

pipeline_builder
::pipeline_builder(boost::program_options::variables_map const& vm, boost::program_options::options_description const& desc)
{
  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline not set" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  {
    std::istream* pistr;
    std::ifstream fin;

    vistk::path_t const ipath = vm["pipeline"].as<vistk::path_t>();

    if (ipath.native() == vistk::path_t("-"))
    {
      pistr = &std::cin;
    }
    else
    {
      fin.open(ipath.native().c_str());

      if (!fin.good())
      {
        static std::string const reason = "Unable to open input file";

        throw std::runtime_error(reason);
      }

      pistr = &fin;
    }

    std::istream& istr = *pistr;

    /// \todo Include paths?

    load_pipeline(istr);
  }

  load_from_options(vm);
}

pipeline_builder
::pipeline_builder()
{
}

pipeline_builder
::~pipeline_builder()
{
}

void
pipeline_builder
::load_pipeline(std::istream& istr)
{
  m_blocks = vistk::load_pipe_blocks(istr, boost::filesystem::current_path());
}

void
pipeline_builder
::load_from_options(boost::program_options::variables_map const& vm)
{
  // Load supplemental configuration files.
  if (vm.count("config"))
  {
    vistk::paths_t const configs = vm["config"].as<vistk::paths_t>();

    std::for_each(configs.begin(), configs.end(), boost::bind(&pipeline_builder::load_supplement, this, _1));
  }

  // Insert lone setting variables from the command line.
  if (vm.count("setting"))
  {
    std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

    std::for_each(settings.begin(), settings.end(), boost::bind(&pipeline_builder::add_setting, this, _1));
  }
}

void
pipeline_builder
::load_supplement(vistk::path_t const& path)
{
  vistk::pipe_blocks const supplement = vistk::load_pipe_blocks_from_file(path);

  m_blocks.insert(m_blocks.end(), supplement.begin(), supplement.end());
}

void
pipeline_builder
::add_setting(std::string const& setting)
{
  vistk::config_pipe_block block;

  vistk::config_value_t value;

  size_t const split_pos = setting.find(split_str);

  if (split_pos == std::string::npos)
  {
    std::string const reason = "Error: The setting on the command line "
                               "\'" + setting + "\' does not contain "
                               "the \'" + split_str + "\' string which "
                               "separates the key from the value";

    throw std::runtime_error(reason);
  }

  vistk::config::key_t setting_key = setting.substr(0, split_pos);
  vistk::config::value_t setting_value = setting.substr(split_pos + split_str.size());

  vistk::config::keys_t keys;

  if (vistk::config::block_sep.size() != 1)
  {
    static std::string const reason = "Error: The block separator is longer than "
                                      "one character and does not work here (this "
                                      "is a vistk limitation)";

    throw std::runtime_error(reason);
  }

  /// \bug Does not work if (vistk::config::block_sep.size() != 1).
  boost::split(keys, setting_key, boost::is_any_of(vistk::config::block_sep));

  if (keys.size() < 2)
  {
    std::string const reason = "Error: The key in the command line setting "
                               "\'" + setting + "\' does not contain "
                               "at least two keys in its keypath which is "
                               "invalid";

    throw std::runtime_error(reason);
  }

  value.key.key_path.push_back(keys.back());
  value.value = setting_value;

  keys.pop_back();

  block.key = keys;
  block.values.push_back(value);

  m_blocks.push_back(block);
}

vistk::pipeline_t
pipeline_builder
::pipeline() const
{
  return vistk::bake_pipe_blocks(m_blocks);
}

vistk::config_t
pipeline_builder
::config() const
{
  return vistk::extract_configuration(m_blocks);
}

vistk::pipe_blocks
pipeline_builder
::blocks() const
{
  return m_blocks;
}

boost::program_options::options_description
pipeline_common_options()
{
  boost::program_options::options_description desc("Common options");

  desc.add_options()
    ("config,c", boost::program_options::value<vistk::paths_t>()->value_name("FILE"), "supplemental configuration file")
    ("setting,s", boost::program_options::value<std::vector<std::string> >()->value_name("VAR=VALUE"), "additional configuration")
    ("include,I", boost::program_options::value<vistk::paths_t>()->value_name("DIR"), "configuration include path")
  ;

  return desc;
}

boost::program_options::options_description
pipeline_input_options()
{
  boost::program_options::options_description desc("Input options");

  desc.add_options()
    ("pipeline,p", boost::program_options::value<vistk::path_t>()->value_name("FILE"), "pipeline")
  ;

  return desc;
}

boost::program_options::options_description
pipeline_output_options()
{
  boost::program_options::options_description desc("Output options");

  desc.add_options()
    ("output,o", boost::program_options::value<vistk::path_t>()->value_name("FILE")->default_value("-"), "output path")
  ;

  return desc;
}

boost::program_options::options_description
pipeline_run_options()
{
  boost::program_options::options_description desc("Run options");

  desc.add_options()
    ("scheduler,S", boost::program_options::value<vistk::scheduler_registry::type_t>()->value_name("TYPE"), "scheduler type")
  ;

  return desc;
}
