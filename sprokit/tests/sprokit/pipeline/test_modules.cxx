// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

//+ move all of these tests to VPM tests
IMPLEMENT_TEST(load)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();
}

// Test loading plugins multiple times causes no problems.
IMPLEMENT_TEST(multiple_load)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();
  vpm.load_all_plugins();
}

//+ Not sure what this is trying to test
TEST_PROPERTY(ENVIRONMENT, KWIVER_PLUGIN_PATH=@CMAKE_CURRENT_BINARY_DIR@/multiple_load)
IMPLEMENT_TEST(envvar)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  const auto proc_type = sprokit::process::type_t("test");

  sprokit::create_process(proc_type, sprokit::process::name_t());

  sprokit::scheduler::type_t const sched_type = sprokit::scheduler::type_t("test");

  sprokit::pipeline_t const pipeline = std::make_shared<sprokit::pipeline>();

  sprokit::create_scheduler(sched_type, pipeline);
}

TEST_PROPERTY(ENVIRONMENT, KWIVER_PLUGIN_PATH=@CMAKE_CURRENT_BINARY_DIR@/not_a_plugin)
IMPLEMENT_TEST(not_a_plugin)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.reload_plugins();
}

TEST_PROPERTY(ENVIRONMENT, KWIVER_PLUGIN_PATH=@CMAKE_CURRENT_BINARY_DIR@)
IMPLEMENT_TEST(has_directory)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.reload_plugins();
}
