/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/scheduler.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST(null_config);
DECLARE_TEST(null_process);
DECLARE_TEST(add_process);
DECLARE_TEST(add_cluster);
DECLARE_TEST(duplicate_process_process);
DECLARE_TEST(connect_no_upstream);
DECLARE_TEST(connect_no_downstream);
DECLARE_TEST(connect_untyped_data_connection);
DECLARE_TEST(connect_untyped_flow_connection);
DECLARE_TEST(connect_type_mismatch);
DECLARE_TEST(connect_flag_shared_no_mutate);
DECLARE_TEST(connect_flag_mismatch_const_mutate);
DECLARE_TEST(connect_flag_mismatch_shared_mutate_first);
DECLARE_TEST(connect_flag_mismatch_shared_mutate_second);
DECLARE_TEST(connect);
DECLARE_TEST(setup_pipeline_no_processes);
DECLARE_TEST(setup_pipeline_orphaned_process);
DECLARE_TEST(setup_pipeline_type_force_flow_upstream);
DECLARE_TEST(setup_pipeline_type_force_flow_downstream);
DECLARE_TEST(setup_pipeline_type_force_cascade_up);
DECLARE_TEST(setup_pipeline_type_force_cascade_down);
DECLARE_TEST(setup_pipeline_type_force_cascade_both);
DECLARE_TEST(setup_pipeline_backwards_edge);
DECLARE_TEST(setup_pipeline_not_a_dag);
DECLARE_TEST(setup_pipeline_data_dependent_set);
DECLARE_TEST(setup_pipeline_data_dependent_set_reject);
DECLARE_TEST(setup_pipeline_data_dependent_set_cascade);
DECLARE_TEST(setup_pipeline_data_dependent_set_cascade_reject);
DECLARE_TEST(setup_pipeline_type_force_flow_upstream_reject);
DECLARE_TEST(setup_pipeline_type_force_flow_downstream_reject);
DECLARE_TEST(setup_pipeline_type_force_cascade_reject);
DECLARE_TEST(setup_pipeline_untyped_data_dependent);
DECLARE_TEST(setup_pipeline_untyped_connection);
DECLARE_TEST(setup_pipeline_missing_required_input_connection);
DECLARE_TEST(setup_pipeline_missing_required_output_connection);
DECLARE_TEST(setup_pipeline_duplicate);
DECLARE_TEST(setup_pipeline_add_process);
DECLARE_TEST(setup_pipeline_connect);
DECLARE_TEST(setup_pipeline);
DECLARE_TEST(start_before_setup);
DECLARE_TEST(start_unsuccessful_setup);
DECLARE_TEST(start_and_stop);
DECLARE_TEST(reset_while_running);
DECLARE_TEST(reset);
DECLARE_TEST(remove_process);
DECLARE_TEST(remove_process_after_setup);
DECLARE_TEST(disconnect);
DECLARE_TEST(disconnect_after_setup);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, null_config);
  ADD_TEST(tests, null_process);
  ADD_TEST(tests, add_process);
  ADD_TEST(tests, add_cluster);
  ADD_TEST(tests, duplicate_process_process);
  ADD_TEST(tests, connect_no_upstream);
  ADD_TEST(tests, connect_no_downstream);
  ADD_TEST(tests, connect_untyped_data_connection);
  ADD_TEST(tests, connect_untyped_flow_connection);
  ADD_TEST(tests, connect_type_mismatch);
  ADD_TEST(tests, connect_flag_shared_no_mutate);
  ADD_TEST(tests, connect_flag_mismatch_const_mutate);
  ADD_TEST(tests, connect_flag_mismatch_shared_mutate_first);
  ADD_TEST(tests, connect_flag_mismatch_shared_mutate_second);
  ADD_TEST(tests, connect);
  ADD_TEST(tests, setup_pipeline_no_processes);
  ADD_TEST(tests, setup_pipeline_orphaned_process);
  ADD_TEST(tests, setup_pipeline_type_force_flow_upstream);
  ADD_TEST(tests, setup_pipeline_type_force_flow_downstream);
  ADD_TEST(tests, setup_pipeline_type_force_cascade_up);
  ADD_TEST(tests, setup_pipeline_type_force_cascade_down);
  ADD_TEST(tests, setup_pipeline_type_force_cascade_both);
  ADD_TEST(tests, setup_pipeline_backwards_edge);
  ADD_TEST(tests, setup_pipeline_not_a_dag);
  ADD_TEST(tests, setup_pipeline_data_dependent_set);
  ADD_TEST(tests, setup_pipeline_data_dependent_set_reject);
  ADD_TEST(tests, setup_pipeline_data_dependent_set_cascade);
  ADD_TEST(tests, setup_pipeline_data_dependent_set_cascade_reject);
  ADD_TEST(tests, setup_pipeline_type_force_flow_upstream_reject);
  ADD_TEST(tests, setup_pipeline_type_force_flow_downstream_reject);
  ADD_TEST(tests, setup_pipeline_type_force_cascade_reject);
  ADD_TEST(tests, setup_pipeline_untyped_data_dependent);
  ADD_TEST(tests, setup_pipeline_untyped_connection);
  ADD_TEST(tests, setup_pipeline_missing_required_input_connection);
  ADD_TEST(tests, setup_pipeline_missing_required_output_connection);
  ADD_TEST(tests, setup_pipeline_duplicate);
  ADD_TEST(tests, setup_pipeline_add_process);
  ADD_TEST(tests, setup_pipeline_connect);
  ADD_TEST(tests, setup_pipeline);
  ADD_TEST(tests, start_before_setup);
  ADD_TEST(tests, start_unsuccessful_setup);
  ADD_TEST(tests, start_and_stop);
  ADD_TEST(tests, reset_while_running);
  ADD_TEST(tests, reset);
  ADD_TEST(tests, remove_process);
  ADD_TEST(tests, remove_process_after_setup);
  ADD_TEST(tests, disconnect);
  ADD_TEST(tests, disconnect_after_setup);

  RUN_TEST(tests, testname);
}

static vistk::process_t create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config = vistk::config::empty_config());
static vistk::pipeline_t create_pipeline();

IMPLEMENT_TEST(null_config)
{
  vistk::config_t const config;

  EXPECT_EXCEPTION(vistk::null_pipeline_config_exception,
                   boost::make_shared<vistk::pipeline>(config),
                   "passing a NULL config to the pipeline");
}

IMPLEMENT_TEST(null_process)
{
  vistk::process_t const process;

  vistk::pipeline_t const pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::null_process_addition_exception,
                   pipeline->add_process(process),
                   "adding a NULL process to the pipeline");
}

IMPLEMENT_TEST(add_process)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
}

IMPLEMENT_TEST(add_cluster)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("multiplier_cluster");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  vistk::process::names_t const names = pipeline->process_names();

  if (names.size() != 2)
  {
    TEST_ERROR("Improperly adding clusters to the pipeline");
  }
}

IMPLEMENT_TEST(duplicate_process_process)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const dup_process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   pipeline->add_process(dup_process),
                   "adding a duplicate process to the pipeline");
}

IMPLEMENT_TEST(connect_no_upstream)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  vistk::process::name_t const proc_name2 = vistk::process::name_t("othername");

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->connect(proc_name2, vistk::process::port_t(),
                                     proc_name, vistk::process::port_t()),
                   "connecting with a non-existent upstream");
}

IMPLEMENT_TEST(connect_no_downstream)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  vistk::process::name_t const proc_name2 = vistk::process::name_t("othername");

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->connect(proc_name, vistk::process::port_t(),
                                     proc_name2, vistk::process::port_t()),
                   "connecting with a non-existent downstream");
}

IMPLEMENT_TEST(connect_untyped_data_connection)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
}

IMPLEMENT_TEST(connect_untyped_flow_connection)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("up");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("down");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
}

IMPLEMENT_TEST(connect_type_mismatch)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typed = vistk::process::type_t("take_string");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("string");

  EXPECT_EXCEPTION(vistk::connection_type_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_named, port_named),
                   "connecting type-mismatched ports");
}

IMPLEMENT_TEST(connect_flag_shared_no_mutate)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("shared");
  vistk::process::type_t const proc_typed = vistk::process::type_t("sink");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named1 = vistk::process::name_t("downstream1");
  vistk::process::name_t const proc_named2 = vistk::process::name_t("downstream2");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd1 = create_process(proc_typed, proc_named1);
  vistk::process_t const processd2 = create_process(proc_typed, proc_named2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd1);
  pipeline->add_process(processd2);

  vistk::process::port_t const port_nameu = vistk::process::port_t("shared");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named1, port_named);
  pipeline->connect(proc_nameu, port_nameu,
                    proc_named2, port_named);
}

IMPLEMENT_TEST(connect_flag_mismatch_const_mutate)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("const");
  vistk::process::type_t const proc_typed = vistk::process::type_t("mutate");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("const");
  vistk::process::port_t const port_named = vistk::process::port_t("mutate");

  EXPECT_EXCEPTION(vistk::connection_flag_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_named, port_named),
                   "connecting a const to a mutate port");
}

IMPLEMENT_TEST(connect_flag_mismatch_shared_mutate_first)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("shared");
  vistk::process::type_t const proc_typed = vistk::process::type_t("sink");
  vistk::process::type_t const proc_typem = vistk::process::type_t("mutate");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namem = vistk::process::name_t("downstream_mutate");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);
  vistk::process_t const processm = create_process(proc_typem, proc_namem);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);
  pipeline->add_process(processm);

  vistk::process::port_t const port_nameu = vistk::process::port_t("shared");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");
  vistk::process::port_t const port_namem = vistk::process::port_t("mutate");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_namem, port_namem);

  EXPECT_EXCEPTION(vistk::connection_flag_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_named, port_named),
                   "connecting to a shared port already connected to a mutate port");
}

IMPLEMENT_TEST(connect_flag_mismatch_shared_mutate_second)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("shared");
  vistk::process::type_t const proc_typed = vistk::process::type_t("sink");
  vistk::process::type_t const proc_typem = vistk::process::type_t("mutate");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namem = vistk::process::name_t("downstream_mutate");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);
  vistk::process_t const processm = create_process(proc_typem, proc_namem);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);
  pipeline->add_process(processm);

  vistk::process::port_t const port_nameu = vistk::process::port_t("shared");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");
  vistk::process::port_t const port_namem = vistk::process::port_t("mutate");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);

  EXPECT_EXCEPTION(vistk::connection_flag_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_namem, port_namem),
                   "connecting a mutate port to a shared port already connected to a port");
}

IMPLEMENT_TEST(connect)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typed = vistk::process::type_t("multiplication");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("factor1");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);
}

IMPLEMENT_TEST(setup_pipeline_no_processes)
{
  vistk::pipeline_t const pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::no_processes_exception,
                   pipeline->setup_pipeline(),
                   "setting up an empty pipeline");
}

IMPLEMENT_TEST(setup_pipeline_orphaned_process)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name1 = vistk::process::name_t("orphan1");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("orphan2");

  vistk::process_t const process1 = create_process(proc_type, proc_name1);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process1);
  pipeline->add_process(process2);

  EXPECT_EXCEPTION(vistk::orphaned_processes_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with orphaned processes");
}

IMPLEMENT_TEST(setup_pipeline_type_force_flow_upstream)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("string");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();
}

IMPLEMENT_TEST(setup_pipeline_type_force_flow_downstream)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("number");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();
}

IMPLEMENT_TEST(setup_pipeline_type_force_cascade_up)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");
  vistk::process::port_t const port_name3 = vistk::process::port_t("string");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
  pipeline->connect(proc_name2, port_name,
                    proc_name3, port_name3);

  pipeline->setup_pipeline();

  vistk::process::port_info_t const info = process->output_port_info(port_name);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly up the pipeline");
  }
}

IMPLEMENT_TEST(setup_pipeline_type_force_cascade_down)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow2");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("number");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");
  vistk::process::port_t const port_name3 = vistk::process::port_t("output");

  pipeline->connect(proc_name2, port_name3,
                    proc_name3, port_name2);
  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();

  vistk::process::port_info_t const info = process3->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly down the pipeline");
  }
}

IMPLEMENT_TEST(setup_pipeline_type_force_cascade_both)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow3");
  vistk::process::name_t const proc_name4 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type, proc_name3);
  vistk::process_t const process4 = create_process(proc_type2, proc_name4);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);
  pipeline->add_process(process4);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");
  vistk::process::port_t const port_name3 = vistk::process::port_t("string");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
  pipeline->connect(proc_name2, port_name,
                    proc_name3, port_name2);
  pipeline->connect(proc_name2, port_name,
                    proc_name4, port_name3);

  pipeline->setup_pipeline();

  vistk::process::port_info_t info;

  info = process->output_port_info(port_name);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly within the pipeline");
  }

  info = process3->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly within the pipeline");
  }
}

IMPLEMENT_TEST(setup_pipeline_backwards_edge)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("feedback");

  vistk::process::name_t const proc_name = vistk::process::name_t("feedback");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name, port_name2);

  pipeline->setup_pipeline();
}

IMPLEMENT_TEST(setup_pipeline_not_a_dag)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("multiplication");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("mult");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");
  vistk::process::port_t const port_name3 = vistk::process::port_t("factor1");
  vistk::process::port_t const port_name4 = vistk::process::port_t("factor2");
  vistk::process::port_t const port_name5 = vistk::process::port_t("product");

  pipeline->connect(proc_name, port_name,
                    proc_name3, port_name3);
  pipeline->connect(proc_name2, port_name,
                    proc_name3, port_name4);
  pipeline->connect(proc_name3, port_name5,
                    proc_name, port_name2);
  pipeline->connect(proc_name3, port_name5,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::not_a_dag_exception,
                   pipeline->setup_pipeline(),
                   "a cycle is in the pipeline graph");
}

IMPLEMENT_TEST(setup_pipeline_data_dependent_set)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();

  vistk::process::port_info_t const info = process2->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly down the pipeline after initialization");
  }
}

IMPLEMENT_TEST(setup_pipeline_data_dependent_set_reject)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2, conf);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_exception,
                   pipeline->setup_pipeline(),
                   "a data dependent type propagation gets rejected");
}

IMPLEMENT_TEST(setup_pipeline_data_dependent_set_cascade)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow2");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
  pipeline->connect(proc_name2, port_name,
                    proc_name3, port_name2);

  pipeline->setup_pipeline();

  vistk::process::port_info_t info;

  info = process2->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly down the pipeline after initialization");
  }

  info = process3->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propagated properly down the pipeline after initialization");
  }
}

IMPLEMENT_TEST(setup_pipeline_data_dependent_set_cascade_reject)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow_reject");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3, conf);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
  pipeline->connect(proc_name2, port_name,
                    proc_name3, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_cascade_exception,
                   pipeline->setup_pipeline(),
                   "a data dependent type propagation gets rejected");
}

IMPLEMENT_TEST(setup_pipeline_type_force_flow_upstream_reject)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("take");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name, conf);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("string");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline where an upstream dependent type that gets rejected");
}

IMPLEMENT_TEST(setup_pipeline_type_force_flow_downstream_reject)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2, conf);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("number");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with a downstream dependent type that gets rejected");
}

IMPLEMENT_TEST(setup_pipeline_type_force_cascade_reject)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow_reject");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3, conf);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);
  pipeline->add_process(process3);

  vistk::process::port_t const port_name = vistk::process::port_t("number");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");
  vistk::process::port_t const port_name3 = vistk::process::port_t("output");

  pipeline->connect(proc_name2, port_name3,
                    proc_name3, port_name2);

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_cascade_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline where a dependent type that gets rejected elsewhere");
}

IMPLEMENT_TEST(setup_pipeline_untyped_data_dependent)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("data_dependent");
  vistk::process::type_t const proc_type2 = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("set_on_configure", "false");

  vistk::process_t const process = create_process(proc_type, proc_name, conf);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::untyped_data_dependent_exception,
                   pipeline->setup_pipeline(),
                   "a connected, unresolved data-dependent port exists after initialization");
}

IMPLEMENT_TEST(setup_pipeline_untyped_connection)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("up");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("down");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::untyped_connection_exception,
                   pipeline->setup_pipeline(),
                   "an untyped connection exists in the pipeline");
}

IMPLEMENT_TEST(setup_pipeline_missing_required_input_connection)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("take_string");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with missing required input connections");
}

IMPLEMENT_TEST(setup_pipeline_missing_required_output_connection)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with missing required output connections");
}

IMPLEMENT_TEST(setup_pipeline_duplicate)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  EXPECT_EXCEPTION(vistk::pipeline_duplicate_setup_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline multiple times");
}

IMPLEMENT_TEST(setup_pipeline_add_process)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  EXPECT_EXCEPTION(vistk::add_after_setup_exception,
                   pipeline->add_process(process),
                   "adding a process after setup");
}

IMPLEMENT_TEST(setup_pipeline_connect)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typed = vistk::process::type_t("sink");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("number");
  vistk::process::name_t const proc_named = vistk::process::name_t("sink");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);

  pipeline->setup_pipeline();

  vistk::process::port_t const iport_name = vistk::process::port_t("status");
  vistk::process::port_t const oport_name = vistk::process::port_heartbeat;

  EXPECT_EXCEPTION(vistk::connection_after_setup_exception,
                   pipeline->connect(proc_named, oport_name,
                                     proc_nameu, iport_name),
                   "making a connection after setup");
}

IMPLEMENT_TEST(setup_pipeline)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typed = vistk::process::type_t("multiplication");
  vistk::process::type_t const proc_typet = vistk::process::type_t("print_number");

  vistk::process::name_t const proc_nameu1 = vistk::process::name_t("upstream1");
  vistk::process::name_t const proc_nameu2 = vistk::process::name_t("upstream2");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  vistk::config_t const configt = vistk::config::empty_config();

  vistk::config::key_t const output_key = vistk::config::key_t("output");
  vistk::config::value_t const output_path = vistk::config::value_t("test-pipeline-setup_pipeline-print_number.txt");

  configt->set_value(output_key, output_path);

  vistk::process_t const processu1 = create_process(proc_typeu, proc_nameu1);
  vistk::process_t const processu2 = create_process(proc_typeu, proc_nameu2);
  vistk::process_t const processd = create_process(proc_typed, proc_named);
  vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu1);
  pipeline->add_process(processu2);
  pipeline->add_process(processd);
  pipeline->add_process(processt);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named1 = vistk::process::port_t("factor1");
  vistk::process::port_t const port_named2 = vistk::process::port_t("factor2");
  vistk::process::port_t const port_namedo = vistk::process::port_t("product");
  vistk::process::port_t const port_namet = vistk::process::port_t("number");

  pipeline->connect(proc_nameu1, port_nameu,
                    proc_named, port_named1);
  pipeline->connect(proc_nameu2, port_nameu,
                    proc_named, port_named2);
  pipeline->connect(proc_named, port_namedo,
                    proc_namet, port_namet);

  pipeline->setup_pipeline();
}

static vistk::scheduler_t create_scheduler(vistk::pipeline_t const& pipe);

IMPLEMENT_TEST(start_before_setup)
{
  vistk::pipeline_t const pipeline = create_pipeline();
  vistk::scheduler_t const scheduler = create_scheduler(pipeline);

  EXPECT_EXCEPTION(vistk::pipeline_not_setup_exception,
                   scheduler->start(),
                   "starting a pipeline that has not been setup");
}

IMPLEMENT_TEST(start_unsuccessful_setup)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("orphan2");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  try
  {
    pipeline->setup_pipeline();
  }
  catch (vistk::pipeline_exception const&)
  {
  }

  vistk::scheduler_t const scheduler = create_scheduler(pipeline);

  EXPECT_EXCEPTION(vistk::pipeline_not_ready_exception,
                   scheduler->start(),
                   "starting a pipeline that has not been successfully setup");
}

IMPLEMENT_TEST(start_and_stop)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  vistk::scheduler_t const scheduler = create_scheduler(pipeline);

  scheduler->start();
  scheduler->stop();
}

IMPLEMENT_TEST(reset_while_running)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  vistk::scheduler_t const scheduler = create_scheduler(pipeline);

  scheduler->start();

  EXPECT_EXCEPTION(vistk::reset_running_pipeline_exception,
                   pipeline->reset(),
                   "resetting a running pipeline");
}

IMPLEMENT_TEST(reset)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typed = vistk::process::type_t("multiplication");
  vistk::process::type_t const proc_typet = vistk::process::type_t("print_number");

  vistk::process::name_t const proc_nameu1 = vistk::process::name_t("upstream1");
  vistk::process::name_t const proc_nameu2 = vistk::process::name_t("upstream2");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  vistk::config_t const configt = vistk::config::empty_config();

  vistk::config::key_t const output_key = vistk::config::key_t("output");
  vistk::config::value_t const output_path = vistk::config::value_t("test-pipeline-setup_pipeline-print_number.txt");

  configt->set_value(output_key, output_path);

  vistk::process_t const processu1 = create_process(proc_typeu, proc_nameu1);
  vistk::process_t const processu2 = create_process(proc_typeu, proc_nameu2);
  vistk::process_t const processd = create_process(proc_typed, proc_named);
  vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

  vistk::pipeline_t const pipeline = create_pipeline();

  pipeline->add_process(processu1);
  pipeline->add_process(processu2);
  pipeline->add_process(processd);
  pipeline->add_process(processt);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named1 = vistk::process::port_t("factor1");
  vistk::process::port_t const port_named2 = vistk::process::port_t("factor2");
  vistk::process::port_t const port_namedo = vistk::process::port_t("product");
  vistk::process::port_t const port_namet = vistk::process::port_t("number");

  pipeline->connect(proc_nameu1, port_nameu,
                    proc_named, port_named1);
  pipeline->connect(proc_nameu2, port_nameu,
                    proc_named, port_named2);
  pipeline->connect(proc_named, port_namedo,
                    proc_namet, port_namet);

  pipeline->setup_pipeline();

  pipeline->reset();

  pipeline->setup_pipeline();
}

IMPLEMENT_TEST(remove_process)
{
  vistk::process::type_t const typeu = vistk::process::type_t("orphan");
  vistk::process::type_t const typed = vistk::process::type_t("sink");
  vistk::process::name_t const nameu = vistk::process::name_t("up");
  vistk::process::name_t const named = vistk::process::name_t("down");

  vistk::pipeline_t const pipe = create_pipeline();
  vistk::process_t const procu = create_process(typeu, nameu);
  vistk::process_t const procd = create_process(typed, named);

  pipe->add_process(procu);
  pipe->add_process(procd);

  vistk::process::port_t const portu = vistk::process::port_heartbeat;
  vistk::process::port_t const portd = vistk::process::port_t("sink");

  pipe->connect(nameu, portu,
                named, portd);

  pipe->remove_process(named);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipe->process_by_name(named),
                   "requesting a process after it has been removed");

  if (!pipe->connections_from_addr(nameu, portu).empty())
  {
    TEST_ERROR("A connection exists after one of the processes has been removed");
  }
}

IMPLEMENT_TEST(remove_process_after_setup)
{
  vistk::process::type_t const type = vistk::process::type_t("orphan");
  vistk::process::name_t const name = vistk::process::name_t("name");

  vistk::pipeline_t const pipe = create_pipeline();
  vistk::process_t const proc = create_process(type, name);

  pipe->add_process(proc);

  pipe->setup_pipeline();

  EXPECT_EXCEPTION(vistk::remove_after_setup_exception,
                   pipe->remove_process(name),
                   "removing a process after the pipeline has been setup");
}

IMPLEMENT_TEST(disconnect)
{
  vistk::process::type_t const typeu = vistk::process::type_t("orphan");
  vistk::process::type_t const typed = vistk::process::type_t("sink");
  vistk::process::name_t const nameu = vistk::process::name_t("up");
  vistk::process::name_t const named = vistk::process::name_t("down");

  vistk::pipeline_t const pipe = create_pipeline();
  vistk::process_t const procu = create_process(typeu, nameu);
  vistk::process_t const procd = create_process(typed, named);

  pipe->add_process(procu);
  pipe->add_process(procd);

  vistk::process::port_t const portu = vistk::process::port_heartbeat;
  vistk::process::port_t const portd = vistk::process::port_t("sink");

  pipe->connect(nameu, portu,
                named, portd);
  pipe->disconnect(nameu, portu,
                   named, portd);

  if (!pipe->connections_from_addr(nameu, portu).empty())
  {
    TEST_ERROR("A connection exists after being disconnected");
  }
}

IMPLEMENT_TEST(disconnect_after_setup)
{
  vistk::process::type_t const typeu = vistk::process::type_t("orphan");
  vistk::process::type_t const typed = vistk::process::type_t("sink");
  vistk::process::name_t const nameu = vistk::process::name_t("up");
  vistk::process::name_t const named = vistk::process::name_t("down");

  vistk::pipeline_t const pipe = create_pipeline();
  vistk::process_t const procu = create_process(typeu, nameu);
  vistk::process_t const procd = create_process(typed, named);

  pipe->add_process(procu);
  pipe->add_process(procd);

  vistk::process::port_t const portu = vistk::process::port_heartbeat;
  vistk::process::port_t const portd = vistk::process::port_t("sink");

  pipe->connect(nameu, portu,
                named, portd);

  pipe->setup_pipeline();

  EXPECT_EXCEPTION(vistk::disconnection_after_setup_exception,
                   pipe->disconnect(nameu, portu,
                                    named, portd),
                   "requesting a disconnect after the pipeline has been setup");
}

vistk::process_t
create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config)
{
  static bool const modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name, config);
}

vistk::pipeline_t
create_pipeline()
{
  return boost::make_shared<vistk::pipeline>();
}

class dummy_scheduler
  : public vistk::scheduler
{
  public:
    dummy_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~dummy_scheduler();

    void _start();
    void _wait();
    void _pause();
    void _resume();
    void _stop();
};

vistk::scheduler_t
create_scheduler(vistk::pipeline_t const& pipe)
{
  vistk::config_t const config = vistk::config::empty_config();

  return boost::make_shared<dummy_scheduler>(pipe, config);
}

dummy_scheduler
::dummy_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config)
  : vistk::scheduler(pipe, config)
{
}

dummy_scheduler
::~dummy_scheduler()
{
}

void
dummy_scheduler
::_start()
{
}

void
dummy_scheduler
::_wait()
{
}

void
dummy_scheduler
::_pause()
{
}

void
dummy_scheduler
::_resume()
{
}

void
dummy_scheduler
::_stop()
{
}
