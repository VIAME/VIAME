/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline_util/export_dot.h>
#include <vistk/pipeline_util/export_dot_exception.h>
#include <vistk/pipeline_util/pipe_bakery.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_cluster.h>
#include <vistk/pipeline/pipeline_exception.h>

#include <boost/make_shared.hpp>

#include <sstream>

#define TEST_ARGS (vistk::path_t const& pipe_file)

DECLARE_TEST(pipeline_null);
DECLARE_TEST(simple_pipeline);
DECLARE_TEST(simple_pipeline_setup);
DECLARE_TEST(simple_pipeline_cluster);
DECLARE_TEST(cluster_null);
DECLARE_TEST(cluster_multiplier);

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  vistk::path_t const pipe_dir = argv[2];

  vistk::path_t const pipe_file = pipe_dir / (testname + pipe_ext);

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, pipeline_null);
  ADD_TEST(tests, simple_pipeline);
  ADD_TEST(tests, simple_pipeline_setup);
  ADD_TEST(tests, simple_pipeline_cluster);
  ADD_TEST(tests, cluster_null);
  ADD_TEST(tests, cluster_multiplier);

  RUN_TEST(tests, testname, pipe_file);
}

IMPLEMENT_TEST(pipeline_null)
{
  (void)pipe_file;

  vistk::pipeline_t const pipeline;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(vistk::null_pipeline_export_dot_exception,
                   vistk::export_dot(sstr, pipeline, "(unnamed)"),
                   "exporting a NULL pipeline to dot");
}

IMPLEMENT_TEST(simple_pipeline)
{
  vistk::load_known_modules();

  vistk::pipeline_t const pipeline = vistk::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  vistk::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(simple_pipeline_setup)
{
  vistk::load_known_modules();

  vistk::pipeline_t const pipeline = vistk::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  vistk::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(simple_pipeline_cluster)
{
  vistk::load_known_modules();

  vistk::pipeline_t const pipeline = vistk::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  vistk::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(cluster_null)
{
  (void)pipe_file;

  vistk::process_cluster_t const cluster;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(vistk::null_cluster_export_dot_exception,
                   vistk::export_dot(sstr, cluster, "(unnamed)"),
                   "exporting a NULL cluster to dot");
}

IMPLEMENT_TEST(cluster_multiplier)
{
  vistk::load_known_modules();

  vistk::cluster_info_t const info = vistk::bake_cluster_from_file(pipe_file);
  vistk::config_t const conf = vistk::config::empty_config();
  vistk::process::name_t const name = vistk::process::name_t("name");

  conf->set_value(vistk::process::config_name, name);

  vistk::process_t const proc = info->ctor(conf);
  vistk::process_cluster_t const cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(proc);

  std::ostringstream sstr;

  vistk::export_dot(sstr, cluster, "(unnamed)");
}
