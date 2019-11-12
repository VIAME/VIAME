/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/vital_types.h>

#include <sprokit/pipeline_util/export_dot.h>
#include <sprokit/pipeline_util/export_dot_exception.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipeline_builder.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/pipeline_exception.h>

#include <memory>
#include <sstream>

#define TEST_ARGS (kwiver::vital::path_t const& pipe_file)

DECLARE_TEST_MAP();

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  kwiver::vital::path_t const pipe_dir = argv[2];

  kwiver::vital::path_t const pipe_file = pipe_dir + "/" +  testname + pipe_ext;

  RUN_TEST(testname, pipe_file);
}


// ----------------------------------------------------------------------------
sprokit::pipeline_t
bake_pipe_from_file( kwiver::vital::path_t const& fname )
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( fname );
  return builder.pipeline();
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(pipeline_null)
{
  (void)pipe_file;

  sprokit::pipeline_t const pipeline;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::null_pipeline_export_dot_exception,
                   sprokit::export_dot(sstr, pipeline, "(unnamed)"),
                   "exporting a NULL pipeline to dot");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(pipeline_empty_name)
{
  (void)pipe_file;

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::process::type_t const type = sprokit::process::type_t("orphan");
  sprokit::process_t const proc = sprokit::create_process(type, sprokit::process::name_t());

  sprokit::pipeline_t const pipe = std::make_shared<sprokit::pipeline>();

  pipe->add_process(proc);

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::empty_name_export_dot_exception,
                   sprokit::export_dot(sstr, pipe, "(unnamed)"),
                   "exporting a pipeline with an empty name to dot");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(simple_pipeline)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_t const pipeline = bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(simple_pipeline_setup)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_t const pipeline = bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(simple_pipeline_cluster)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_t const pipeline = bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, pipeline, "(unnamed)");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_null)
{
  (void)pipe_file;

  sprokit::process_cluster_t const cluster;

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::null_cluster_export_dot_exception,
                   sprokit::export_dot(sstr, cluster, "(unnamed)"),
                   "exporting a NULL cluster to dot");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_empty_name)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_builder builder;
  builder.load_cluster( pipe_file );
  sprokit::cluster_info_t const info = builder.cluster_info();
  const auto conf = kwiver::vital::config_block::empty_config();

  sprokit::process_t const proc = info->ctor(conf);
  sprokit::process_cluster_t const cluster = std::dynamic_pointer_cast<sprokit::process_cluster>(proc);

  std::ostringstream sstr;

  EXPECT_EXCEPTION(sprokit::empty_name_export_dot_exception,
                   sprokit::export_dot(sstr, cluster, "(unnamed)"),
                   "exporting a cluster with an empty name to dot");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_multiplier)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_builder builder;
  builder.load_cluster( pipe_file );
  sprokit::cluster_info_t const info = builder.cluster_info();
  const auto conf = kwiver::vital::config_block::empty_config();
  sprokit::process::name_t const name = sprokit::process::name_t("name");

  conf->set_value(sprokit::process::config_name, name);

  sprokit::process_t const proc = info->ctor(conf);
  sprokit::process_cluster_t const cluster = std::dynamic_pointer_cast<sprokit::process_cluster>(proc);

  std::ostringstream sstr;

  sprokit::export_dot(sstr, cluster, "(unnamed)");
}
