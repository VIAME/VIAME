/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <sprokit/pipeline_util/export_dot.h>
#include <sprokit/pipeline_util/export_dot_exception.h>
#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/pipeline_exception.h>

#include <boost/make_shared.hpp>

#include <sstream>

#define TEST_ARGS (sprokit::path_t const& pipe_file)

DECLARE_TEST(pipeline_null);
DECLARE_TEST(pipeline_empty_name);
DECLARE_TEST(simple_pipeline);
DECLARE_TEST(simple_pipeline_setup);
DECLARE_TEST(simple_pipeline_cluster);
DECLARE_TEST(cluster_null);
DECLARE_TEST(cluster_empty_name);
DECLARE_TEST(cluster_multiplier);

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  sprokit::path_t const pipe_dir = argv[2];

  sprokit::path_t const pipe_file = pipe_dir / (testname + pipe_ext);

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, pipeline_null);
  ADD_TEST(tests, pipeline_empty_name);
  ADD_TEST(tests, simple_pipeline);
  ADD_TEST(tests, simple_pipeline_setup);
  ADD_TEST(tests, simple_pipeline_cluster);
  ADD_TEST(tests, cluster_null);
  ADD_TEST(tests, cluster_empty_name);
  ADD_TEST(tests, cluster_multiplier);

  RUN_TEST(tests, testname, pipe_file);
}

IMPLEMENT_TEST(pipeline_null)
{
  (void)pipe_file;

  sprokit::pipeline_t const pipeline;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::null_pipeline_export_dot_exception,
                   sprokit::export_dot(sstr, pipeline, "(unnamed)"),
                   "exporting a NULL pipeline to dot");
}

IMPLEMENT_TEST(pipeline_empty_name)
{
  (void)pipe_file;

  sprokit::load_known_modules();

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const type = sprokit::process::type_t("orphan");
  sprokit::process_t const proc = reg->create_process(type, sprokit::process::name_t());

  sprokit::pipeline_t const pipe = boost::make_shared<sprokit::pipeline>();

  pipe->add_process(proc);

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::empty_name_export_dot_exception,
                   sprokit::export_dot(sstr, pipe, "(unnamed)"),
                   "exporting a pipeline with an empty name to dot");
}

IMPLEMENT_TEST(simple_pipeline)
{
  sprokit::load_known_modules();

  sprokit::pipeline_t const pipeline = sprokit::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(simple_pipeline_setup)
{
  sprokit::load_known_modules();

  sprokit::pipeline_t const pipeline = sprokit::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(simple_pipeline_cluster)
{
  sprokit::load_known_modules();

  sprokit::pipeline_t const pipeline = sprokit::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}

IMPLEMENT_TEST(cluster_null)
{
  (void)pipe_file;

  sprokit::process_cluster_t const cluster;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::null_cluster_export_dot_exception,
                   sprokit::export_dot(sstr, cluster, "(unnamed)"),
                   "exporting a NULL cluster to dot");
}

IMPLEMENT_TEST(cluster_empty_name)
{
  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_from_file(pipe_file);
  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::process_t const proc = info->ctor(conf);
  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(proc);

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::empty_name_export_dot_exception,
                   sprokit::export_dot(sstr, cluster, "(unnamed)"),
                   "exporting a cluster with an empty name to dot");
}

IMPLEMENT_TEST(cluster_multiplier)
{
  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_from_file(pipe_file);
  sprokit::config_t const conf = sprokit::config::empty_config();
  sprokit::process::name_t const name = sprokit::process::name_t("name");

  conf->set_value(sprokit::process::config_name, name);

  sprokit::process_t const proc = info->ctor(conf);
  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(proc);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, cluster, "(unnamed)");
}
