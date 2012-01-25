/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/export_dot.h>
#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/types.h>

#include <vistk/config.h>

#include <tools/helpers/typed_value_desc.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdlib>

namespace po = boost::program_options;

static std::string const split_str = "=";

static po::options_description make_options();
static void VISTK_NO_RETURN usage(po::options_description const& options);

int main(int argc, char* argv[])
{
  vistk::load_known_modules();

  po::options_description const desc = make_options();

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    usage(desc);
  }

  if (!vm.count("input"))
  {
    std::cerr << "Error: input not set" << std::endl;
    usage(desc);
  }

  vistk::pipeline_t pipe;

  {
    std::istream* pistr;
    std::ifstream fin;

    vistk::path_t const ipath = vm["input"].as<vistk::path_t>();

    if (ipath.native() == vistk::path_t("-"))
    {
      pistr = &std::cin;
    }
    else
    {
      fin.open(ipath.native().c_str());

      if (!fin.good())
      {
        std::cerr << "Error: Unable to open input file" << std::endl;

        return 1;
      }

      pistr = &fin;
    }

    std::istream& istr = *pistr;

    /// \todo Include paths?

    vistk::pipe_blocks blocks = vistk::load_pipe_blocks(istr, boost::filesystem::current_path());

    // Load supplemental configuration files.
    if (vm.count("config"))
    {
      vistk::paths_t const configs = vm["config"].as<vistk::paths_t>();

      BOOST_FOREACH (vistk::path_t const& config, configs)
      {
        vistk::pipe_blocks const config_blocks = vistk::load_pipe_blocks_from_file(config);

        blocks.insert(blocks.end(), config_blocks.begin(), config_blocks.end());
      }
    }

    // Insert lone setting variables from the command line.
    if (vm.count("setting"))
    {
      std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

      BOOST_FOREACH (std::string const& setting, settings)
      {
        vistk::config_pipe_block block;

        vistk::config_value_t value;

        size_t const split_pos = setting.find(split_str);

        if (split_pos == std::string::npos)
        {
          std::cerr << "Error: The setting on the command line "
                       "\'" << setting << "\' does not contain "
                       "the \'" << split_str << "\' string which "
                       "separates the key from the value" << std::endl;

          return 1;
        }

        vistk::config::key_t setting_key = setting.substr(0, split_pos);
        vistk::config::value_t setting_value = setting.substr(split_pos + split_str.size());

        vistk::config::keys_t keys;

        if (vistk::config::block_sep.size() != 1)
        {
          std::cerr << "Error: The block separator is not than "
                       "one character and does not work here" << std::endl;

          return 1;
        }

        /// \bug Does not work if (vistk::config::block_sep.size() != 1).
        boost::split(keys, setting_key, boost::is_any_of(vistk::config::block_sep));

        if (keys.size() < 2)
        {
          std::cerr << "Error: The key in the command line setting "
                       "\'" << setting << "\' does not contain "
                       "at least two keys in its keypath which is "
                       "invalid" << std::endl;

          return 1;
        }

        value.key.key_path.push_back(keys.back());
        value.value = setting_value;

        keys.pop_back();

        block.key = keys;
        block.values.push_back(value);

        blocks.push_back(block);
      }
    }

    pipe = vistk::bake_pipe_blocks(blocks);
  }

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return 1;
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

      return 1;
    }

    postr = &fout;
  }

  std::ostream& ostr = *postr;

  std::string const graph_name = vm["name"].as<std::string>();

  vistk::export_dot(ostr, pipe, graph_name);

  return 0;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("input,i", po::value_desc<vistk::path_t>()->metavar("FILE"), "input path")
    ("output,o", po::value_desc<vistk::path_t>()->metavar("FILE")->default_value("-"), "output path")
    ("config,c", po::value_desc<vistk::paths_t>()->metavar("FILE"), "supplemental configuration file")
    ("setting,s", po::value_desc<std::vector<std::string> >()->metavar("VAR=VALUE"), "additional configuration")
    ("include,I", po::value_desc<vistk::paths_t>()->metavar("DIR"), "configuration include path")
    ("name,n", po::value_desc<std::string>()->metavar("NAME")->default_value("unnamed"), "name of the graph")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(1);
}
