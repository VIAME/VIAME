/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline_util/export_dot.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/program_options/variables_map.hpp>
#include <boost/bind.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdlib>

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

  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline not set" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
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

        return EXIT_FAILURE;
      }

      pistr = &fin;
    }

    std::istream& istr = *pistr;

    /// \todo Include paths?

    pipeline_builder builder;

    builder.load_pipeline(istr);

    // Load supplemental configuration files.
    if (vm.count("config"))
    {
      vistk::paths_t const configs = vm["config"].as<vistk::paths_t>();

      std::for_each(configs.begin(), configs.end(), boost::bind(&pipeline_builder::load_supplement, &builder, _1));
    }

    // Insert lone setting variables from the command line.
    if (vm.count("setting"))
    {
      std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

      std::for_each(settings.begin(), settings.end(), boost::bind(&pipeline_builder::add_setting, &builder, _1));
    }

    pipe = builder.pipeline();
  }

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

  std::string const graph_name = vm["name"].as<std::string>();

  if (vm.count("setup"))
  {
    pipe->setup_pipeline();

    vistk::export_dot_setup(ostr, pipe, graph_name);
  }
  else
  {
    vistk::export_dot(ostr, pipe, graph_name);
  }

  return EXIT_SUCCESS;
}
