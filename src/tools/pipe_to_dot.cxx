/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_io.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline_util/export_dot.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_cluster.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <string>
#include <vector>

#include <cstddef>
#include <cstdlib>

static boost::program_options::options_description cluster_options();
static boost::program_options::options_description pipe_to_dot_options();

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(pipeline_common_options())
    .add(pipeline_input_options())
    .add(cluster_options())
    .add(pipeline_output_options())
    .add(pipe_to_dot_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  vistk::process_cluster_t cluster;
  vistk::pipeline_t pipe;

  bool const have_cluster_type = vm.count("cluster-type");
  bool const have_pipeline = vm.count("pipeline");


  if (have_cluster_type && have_pipeline)
  {
    std::cerr << "Error: The and \'cluster-type\' option is incompatible "
                 "with the \'pipeline\' option" << std::endl;

    return EXIT_FAILURE;
  }

  std::string const graph_name = vm["name"].as<std::string>();

  if (have_cluster_type)
  {
    pipeline_builder builder;

    builder.load_from_options(vm);
    vistk::config_t const conf = builder.config();

    vistk::process_registry_t const reg = vistk::process_registry::self();

    vistk::process::type_t const type = vm["cluster-type"].as<vistk::process::type_t>();

    vistk::process_t const proc = reg->create_process(type, graph_name, conf);
    cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(proc);

    if (!cluster)
    {
      std::cerr << "Error: The given type (\'" << type << "\') "
                   "is not a cluster" << std::endl;

      return EXIT_FAILURE;
    }
  }
  else if (have_pipeline)
  {
    pipeline_builder const builder(vm, desc);

    pipe = builder.pipeline();

    if (!pipe)
    {
      std::cerr << "Error: Unable to bake pipeline" << std::endl;

      return EXIT_FAILURE;
    }
  }
  else
  {
    std::cerr << "Error: Either \'cluster-type\' or \'pipeline\' "
                 "must be specified" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  // Make sure we have one...
  if (!cluster && !pipe)
  {
    std::cerr << "Internal error: option tracking failure" << std::endl;

    return EXIT_FAILURE;
  }

  // ...but not both.
  if (cluster && pipe)
  {
    std::cerr << "Internal error: option tracking failure" << std::endl;

    return EXIT_FAILURE;
  }

  vistk::path_t const opath = vm["output"].as<vistk::path_t>();

  ostream_t const ostr = open_ostream(opath);

  if (cluster)
  {
    vistk::export_dot(*ostr, cluster, graph_name);
  }
  else if (pipe)
  {
    if (vm.count("setup"))
    {
      pipe->setup_pipeline();
    }

    vistk::export_dot(*ostr, pipe, graph_name);
  }
  else
  {
    std::cerr << "Internal error: option tracking failure" << std::endl;

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

boost::program_options::options_description
cluster_options()
{
  boost::program_options::options_description desc("Cluster options");

  desc.add_options()
    ("cluster-type,T", boost::program_options::value<std::string>()->value_name("TYPE"), "the cluster type to export")
  ;

  return desc;
}

boost::program_options::options_description
pipe_to_dot_options()
{
  boost::program_options::options_description desc;

  desc.add_options()
    ("name,n", boost::program_options::value<std::string>()->value_name("NAME")->default_value("unnamed"), "the name of the graph")
    ("setup", "whether to setup the pipeline before exporting or not")
  ;

  return desc;
}
