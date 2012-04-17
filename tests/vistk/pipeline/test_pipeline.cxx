/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <boost/algorithm/string/predicate.hpp>
#include <boost/make_shared.hpp>

#include <exception>
#include <iostream>
#include <string>

#include <cstdlib>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_null_config();
static void test_null_process();
static void test_add_process();
static void test_add_group();
static void test_duplicate_process_process();
static void test_duplicate_process_group();
static void test_duplicate_group_process();
static void test_duplicate_group_group();
static void test_map_input_no_group();
static void test_map_output_no_group();
static void test_map_input_no_process();
static void test_map_output_no_process();
static void test_map_input();
static void test_map_output();
static void test_connect_no_upstream();
static void test_connect_no_downstream();
static void test_connect_untyped_data_connection();
static void test_connect_untyped_flow_connection();
static void test_connect_type_mismatch();
static void test_connect_flag_mismatch();
static void test_connect();
static void test_connect_input_map();
static void test_connect_output_map();
static void test_setup_pipeline_no_processes();
static void test_setup_pipeline_orphaned_process();
static void test_setup_pipeline_type_force_flow_upstream();
static void test_setup_pipeline_type_force_flow_downstream();
static void test_setup_pipeline_type_force_cascade_up();
static void test_setup_pipeline_type_force_cascade_down();
static void test_setup_pipeline_type_force_cascade_both();
static void test_setup_pipeline_backwards_edge();
static void test_setup_pipeline_not_a_dag();
static void test_setup_pipeline_data_dependent_set();
static void test_setup_pipeline_data_dependent_set_reject();
static void test_setup_pipeline_data_dependent_set_cascade();
static void test_setup_pipeline_data_dependent_set_cascade_reject();
static void test_setup_pipeline_type_force_flow_upstream_reject();
static void test_setup_pipeline_type_force_flow_downstream_reject();
static void test_setup_pipeline_type_force_cascade_reject();
static void test_setup_pipeline_untyped_data_dependent();
static void test_setup_pipeline_untyped_connection();
static void test_setup_pipeline_missing_required_input_connection();
static void test_setup_pipeline_missing_required_output_connection();
static void test_setup_pipeline_missing_required_group_input_connection();
static void test_setup_pipeline_missing_required_group_output_connection();
static void test_setup_pipeline_duplicate();
static void test_setup_pipeline_add_process();
static void test_setup_pipeline_add_group();
static void test_setup_pipeline_connect();
static void test_setup_pipeline_map_input();
static void test_setup_pipeline_map_output();
static void test_setup_pipeline();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "null_process")
  {
    test_null_process();
  }
  else if (test_name == "add_process")
  {
    test_add_process();
  }
  else if (test_name == "add_group")
  {
    test_add_group();
  }
  else if (test_name == "duplicate_process_process")
  {
    test_duplicate_process_process();
  }
  else if (test_name == "duplicate_process_group")
  {
    test_duplicate_process_group();
  }
  else if (test_name == "duplicate_group_process")
  {
    test_duplicate_group_process();
  }
  else if (test_name == "duplicate_group_group")
  {
    test_duplicate_group_group();
  }
  else if (test_name == "map_input_no_group")
  {
    test_map_input_no_group();
  }
  else if (test_name == "map_output_no_group")
  {
    test_map_output_no_group();
  }
  else if (test_name == "map_input_no_process")
  {
    test_map_input_no_process();
  }
  else if (test_name == "map_output_no_process")
  {
    test_map_output_no_process();
  }
  else if (test_name == "map_input")
  {
    test_map_input();
  }
  else if (test_name == "map_output")
  {
    test_map_output();
  }
  else if (test_name == "connect_no_upstream")
  {
    test_connect_no_upstream();
  }
  else if (test_name == "connect_no_downstream")
  {
    test_connect_no_downstream();
  }
  else if (test_name == "connect_untyped_data_connection")
  {
    test_connect_untyped_data_connection();
  }
  else if (test_name == "connect_untyped_flow_connection")
  {
    test_connect_untyped_flow_connection();
  }
  else if (test_name == "connect_type_mismatch")
  {
    test_connect_type_mismatch();
  }
  else if (test_name == "connect_flag_mismatch")
  {
    test_connect_flag_mismatch();
  }
  else if (test_name == "connect")
  {
    test_connect();
  }
  else if (test_name == "connect_input_map")
  {
    test_connect_input_map();
  }
  else if (test_name == "connect_output_map")
  {
    test_connect_output_map();
  }
  else if (test_name == "setup_pipeline_no_processes")
  {
    test_setup_pipeline_no_processes();
  }
  else if (test_name == "setup_pipeline_orphaned_process")
  {
    test_setup_pipeline_orphaned_process();
  }
  else if (test_name == "setup_pipeline_type_force_flow_upstream")
  {
    test_setup_pipeline_type_force_flow_upstream();
  }
  else if (test_name == "setup_pipeline_type_force_flow_downstream")
  {
    test_setup_pipeline_type_force_flow_downstream();
  }
  else if (test_name == "setup_pipeline_type_force_cascade_up")
  {
    test_setup_pipeline_type_force_cascade_up();
  }
  else if (test_name == "setup_pipeline_type_force_cascade_down")
  {
    test_setup_pipeline_type_force_cascade_down();
  }
  else if (test_name == "setup_pipeline_type_force_cascade_both")
  {
    test_setup_pipeline_type_force_cascade_both();
  }
  else if (test_name == "setup_pipeline_backwards_edge")
  {
    test_setup_pipeline_backwards_edge();
  }
  else if (test_name == "setup_pipeline_not_a_dag")
  {
    test_setup_pipeline_not_a_dag();
  }
  else if (test_name == "setup_pipeline_data_dependent_set")
  {
    test_setup_pipeline_data_dependent_set();
  }
  else if (test_name == "setup_pipeline_data_dependent_set_reject")
  {
    test_setup_pipeline_data_dependent_set_reject();
  }
  else if (test_name == "setup_pipeline_data_dependent_set_cascade")
  {
    test_setup_pipeline_data_dependent_set_cascade();
  }
  else if (test_name == "setup_pipeline_data_dependent_set_cascade_reject")
  {
    test_setup_pipeline_data_dependent_set_cascade_reject();
  }
  else if (test_name == "setup_pipeline_type_force_flow_upstream_reject")
  {
    test_setup_pipeline_type_force_flow_upstream_reject();
  }
  else if (test_name == "setup_pipeline_type_force_flow_downstream_reject")
  {
    test_setup_pipeline_type_force_flow_downstream_reject();
  }
  else if (test_name == "setup_pipeline_type_force_cascade_reject")
  {
    test_setup_pipeline_type_force_cascade_reject();
  }
  else if (test_name == "setup_pipeline_untyped_data_dependent")
  {
    test_setup_pipeline_untyped_data_dependent();
  }
  else if (test_name == "setup_pipeline_untyped_connection")
  {
    test_setup_pipeline_untyped_connection();
  }
  else if (test_name == "setup_pipeline_missing_required_input_connection")
  {
    test_setup_pipeline_missing_required_input_connection();
  }
  else if (test_name == "setup_pipeline_missing_required_output_connection")
  {
    test_setup_pipeline_missing_required_output_connection();
  }
  else if (test_name == "setup_pipeline_missing_required_group_input_connection")
  {
    test_setup_pipeline_missing_required_group_input_connection();
  }
  else if (test_name == "setup_pipeline_missing_required_group_output_connection")
  {
    test_setup_pipeline_missing_required_group_output_connection();
  }
  else if (test_name == "setup_pipeline_duplicate")
  {
    test_setup_pipeline_duplicate();
  }
  else if (test_name == "setup_pipeline_add_process")
  {
    test_setup_pipeline_add_process();
  }
  else if (test_name == "setup_pipeline_add_group")
  {
    test_setup_pipeline_add_group();
  }
  else if (test_name == "setup_pipeline_connect")
  {
    test_setup_pipeline_connect();
  }
  else if (test_name == "setup_pipeline_map_input")
  {
    test_setup_pipeline_map_input();
  }
  else if (test_name == "setup_pipeline_map_output")
  {
    test_setup_pipeline_map_output();
  }
  else if (test_name == "setup_pipeline")
  {
    test_setup_pipeline();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

static vistk::process_t create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name, vistk::config_t config = vistk::config::empty_config());
static vistk::pipeline_t create_pipeline();

void
test_null_config()
{
  vistk::config_t const config;

  EXPECT_EXCEPTION(vistk::null_pipeline_config_exception,
                   boost::make_shared<vistk::pipeline>(config),
                   "passing a NULL config to the pipeline");
}
void
test_null_process()
{
  vistk::process_t const process;

  vistk::pipeline_t pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::null_process_addition_exception,
                   pipeline->add_process(process),
                   "adding a NULL process to the pipeline");
}

void
test_add_process()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
}

void
test_add_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);
}

void
test_duplicate_process_process()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const dup_process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   pipeline->add_process(dup_process),
                   "adding a duplicate process to the pipeline");
}

void
test_duplicate_process_group()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const dup_process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   pipeline->add_process(dup_process),
                   "adding a duplicate process to the pipeline");
}

void
test_duplicate_group_process()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   pipeline->add_group(proc_name),
                   "adding a duplicate group to the pipeline");
}

void
test_duplicate_group_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   pipeline->add_group(proc_name),
                   "adding a duplicate group to the pipeline");
}

void
test_map_input_no_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::no_such_group_exception,
                   pipeline->map_input_port(proc_name, vistk::process::port_t(),
                                            vistk::process::name_t(), vistk::process::port_t(),
                                            vistk::process::port_flags_t()),
                   "mapping an input on a non-existent group");
}

void
test_map_output_no_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::no_such_group_exception,
                   pipeline->map_output_port(proc_name, vistk::process::port_t(),
                                             vistk::process::name_t(), vistk::process::port_t(),
                                             vistk::process::port_flags_t()),
                   "mapping an output to a non-existent group");
}

void
test_map_input_no_process()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);

  pipeline->map_input_port(proc_name, vistk::process::port_t(),
                           vistk::process::name_t(), vistk::process::port_t(),
                           vistk::process::port_flags_t());
}

void
test_map_output_no_process()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);

  pipeline->map_output_port(proc_name, vistk::process::port_t(),
                            vistk::process::name_t(), vistk::process::port_t(),
                            vistk::process::port_flags_t());
}

void
test_map_input()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const group_name = vistk::process::name_t("group");
  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(group_name);
  pipeline->add_process(process);

  pipeline->map_input_port(group_name, vistk::process::port_t(),
                           proc_name, vistk::process::port_t(),
                           vistk::process::port_flags_t());
}

void
test_map_output()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const group_name = vistk::process::name_t("group");
  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  vistk::process::port_t const port_name = vistk::process::port_t("port");

  pipeline->add_group(group_name);
  pipeline->add_process(process);

  pipeline->map_output_port(group_name, port_name,
                            proc_name, vistk::process::port_t(),
                            vistk::process::port_flags_t());

  EXPECT_EXCEPTION(vistk::group_output_already_mapped_exception,
                   pipeline->map_output_port(group_name, port_name,
                                             proc_name, vistk::process::port_t(),
                                             vistk::process::port_flags_t()),
                   "mapping an output on an non-existent group");
}

void
test_connect_no_upstream()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  vistk::process::name_t const proc_name2 = vistk::process::name_t("othername");

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->connect(proc_name2, vistk::process::port_t(),
                                     proc_name, vistk::process::port_t()),
                   "connecting with a non-existent upstream");
}

void
test_connect_no_downstream()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process::name_t const proc_name = vistk::process::name_t("name");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  vistk::process::name_t const proc_name2 = vistk::process::name_t("othername");

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->connect(proc_name, vistk::process::port_t(),
                                     proc_name2, vistk::process::port_t()),
                   "connecting with a non-existent downstream");
}

void
test_connect_untyped_data_connection()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
}

void
test_connect_untyped_flow_connection()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("up");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("down");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);
}

void
test_connect_type_mismatch()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("take_string");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("string");

  EXPECT_EXCEPTION(vistk::connection_type_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_named, port_named),
                   "connecting type-mismatched ports");
}

void
test_connect_flag_mismatch()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("const");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("mutate");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("const");
  vistk::process::port_t const port_named = vistk::process::port_t("mutate");

  EXPECT_EXCEPTION(vistk::connection_flag_mismatch_exception,
                   pipeline->connect(proc_nameu, port_nameu,
                                     proc_named, port_named),
                   "connecting flag-mismatched ports");
}

void
test_connect()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("multiplication");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("factor1");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);
}

void
test_connect_input_map()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("multiplication");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::name_t const group_name = vistk::process::name_t("group");
  vistk::process::port_t const group_port = vistk::process::name_t("mapped_port");

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("factor1");

  pipeline->add_group(group_name);
  pipeline->map_input_port(group_name, group_port,
                           proc_named, port_named,
                           vistk::process::port_flags_t());

  pipeline->connect(proc_nameu, port_nameu,
                    group_name, group_port);
}

void
test_connect_output_map()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("multiplication");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::name_t const group_name = vistk::process::name_t("group");
  vistk::process::port_t const group_port = vistk::process::name_t("mapped_port");

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("factor1");

  pipeline->add_group(group_name);
  pipeline->map_output_port(group_name, group_port,
                            proc_nameu, port_nameu,
                            vistk::process::port_flags_t());

  pipeline->connect(group_name, group_port,
                    proc_named, port_named);
}

void
test_setup_pipeline_no_processes()
{
  vistk::pipeline_t pipeline = create_pipeline();

  EXPECT_EXCEPTION(vistk::no_processes_exception,
                   pipeline->setup_pipeline(),
                   "setting up an empty pipeline");
}

void
test_setup_pipeline_orphaned_process()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process::name_t const proc_name1 = vistk::process::name_t("orphan1");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("orphan2");

  vistk::process_t const process1 = create_process(proc_type, proc_name1);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process1);
  pipeline->add_process(process2);

  EXPECT_EXCEPTION(vistk::orphaned_processes_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with orphaned processes");
}

void
test_setup_pipeline_type_force_flow_upstream()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("string");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();
}

void
test_setup_pipeline_type_force_flow_downstream()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("number");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  pipeline->setup_pipeline();
}

void
test_setup_pipeline_type_force_cascade_up()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t pipeline = create_pipeline();

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
    TEST_ERROR("Dependent types were not propogated properly up the pipeline");
  }
}

void
test_setup_pipeline_type_force_cascade_down()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow2");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t pipeline = create_pipeline();

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
    TEST_ERROR("Dependent types were not propogated properly down the pipeline");
  }
}

void
test_setup_pipeline_type_force_cascade_both()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow3");
  vistk::process::name_t const proc_name4 = vistk::process::name_t("take");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type, proc_name3);
  vistk::process_t const process4 = create_process(proc_type2, proc_name4);

  vistk::pipeline_t pipeline = create_pipeline();

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
    TEST_ERROR("Dependent types were not propogated properly within the pipeline");
  }

  info = process3->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propogated properly within the pipeline");
  }
}

void
test_setup_pipeline_backwards_edge()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("feedback");

  vistk::process::name_t const proc_name = vistk::process::name_t("feedback");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name, port_name2);

  pipeline->setup_pipeline();
}

void
test_setup_pipeline_not_a_dag()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("multiplication");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow2");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("mult");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_data_dependent_set()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

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
    TEST_ERROR("Dependent types were not propogated properly down the pipeline after initialization");
  }
}

void
test_setup_pipeline_data_dependent_set_reject()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("sink");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2, conf);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_process(process2);

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_t const port_name2 = vistk::process::port_t("input");

  pipeline->connect(proc_name, port_name,
                    proc_name2, port_name2);

  EXPECT_EXCEPTION(vistk::connection_dependent_type_exception,
                   pipeline->setup_pipeline(),
                   "a data dependent type propogation gets rejected");
}

void
test_setup_pipeline_data_dependent_set_cascade()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow2");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3);

  vistk::pipeline_t pipeline = create_pipeline();

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
    TEST_ERROR("Dependent types were not propogated properly down the pipeline after initialization");
  }

  info = process3->input_port_info(port_name2);

  if (boost::starts_with(info->type, vistk::process::type_flow_dependent))
  {
    TEST_ERROR("Dependent types were not propogated properly down the pipeline after initialization");
  }
}

void
test_setup_pipeline_data_dependent_set_cascade_reject()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow_reject");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3, conf);

  vistk::pipeline_t pipeline = create_pipeline();

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
                   "a data dependent type propogation gets rejected");
}

void
test_setup_pipeline_type_force_flow_upstream_reject()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("take_string");

  vistk::process::name_t const proc_name = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("take");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name, conf);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_type_force_flow_downstream_reject()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2, conf);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_type_force_cascade_reject()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");
  vistk::process::name_t const proc_name3 = vistk::process::name_t("flow_reject");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("reject", "true");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);
  vistk::process_t const process3 = create_process(proc_type2, proc_name3, conf);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_untyped_data_dependent()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("data_dependent");
  vistk::process_registry::type_t const proc_type2 = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("data");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("flow");

  vistk::config_t conf = vistk::config::empty_config();

  conf->set_value("set_on_configure", "false");

  vistk::process_t const process = create_process(proc_type, proc_name, conf);
  vistk::process_t const process2 = create_process(proc_type2, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_untyped_connection()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

  vistk::process::name_t const proc_name = vistk::process::name_t("up");
  vistk::process::name_t const proc_name2 = vistk::process::name_t("down");

  vistk::process_t const process = create_process(proc_type, proc_name);
  vistk::process_t const process2 = create_process(proc_type, proc_name2);

  vistk::pipeline_t pipeline = create_pipeline();

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

void
test_setup_pipeline_missing_required_input_connection()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("take_string");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with missing required input connections");
}

void
test_setup_pipeline_missing_required_output_connection()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline with missing required output connections");
}

void
test_setup_pipeline_missing_required_group_input_connection()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typet = vistk::process_registry::type_t("print_number");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");
  vistk::process::name_t const group_name = vistk::process::name_t("group");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processt = create_process(proc_typet, proc_namet);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processt);
  pipeline->add_group(group_name);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_namet = vistk::process::port_t("number");
  vistk::process::port_t const group_port = vistk::process::port_t("group_port");

  vistk::process::port_flags_t flags;

  flags.insert(vistk::process::flag_required);

  pipeline->map_input_port(group_name, group_port,
                           proc_namet, port_namet,
                           flags);

  pipeline->connect(proc_nameu, port_nameu,
                    proc_namet, port_namet);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "missing required output port connection");
}

void
test_setup_pipeline_missing_required_group_output_connection()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typet = vistk::process_registry::type_t("print_number");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");
  vistk::process::name_t const group_name = vistk::process::name_t("group");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processt = create_process(proc_typet, proc_namet);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processt);
  pipeline->add_group(group_name);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_namet = vistk::process::port_t("number");
  vistk::process::port_t const group_port = vistk::process::port_t("group_port");

  vistk::process::port_flags_t flags;

  flags.insert(vistk::process::flag_required);

  pipeline->map_output_port(group_name, group_port,
                            proc_nameu, port_nameu,
                            flags);

  pipeline->connect(proc_nameu, port_nameu,
                    proc_namet, port_namet);

  EXPECT_EXCEPTION(vistk::missing_connection_exception,
                   pipeline->setup_pipeline(),
                   "missing required output port connection");
}

void
test_setup_pipeline_duplicate()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  EXPECT_EXCEPTION(vistk::pipeline_duplicate_setup_exception,
                   pipeline->setup_pipeline(),
                   "setting up a pipeline multiple times");
}

void
test_setup_pipeline_add_process()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  EXPECT_EXCEPTION(vistk::add_after_setup_exception,
                   pipeline->add_process(process),
                   "adding a process after setup");
}

void
test_setup_pipeline_add_group()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");
  vistk::process::name_t const group_name = vistk::process::name_t("group");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);

  pipeline->setup_pipeline();

  EXPECT_EXCEPTION(vistk::add_after_setup_exception,
                   pipeline->add_group(group_name),
                   "adding a group after setup");
}

void
test_setup_pipeline_connect()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("sink");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("number");
  vistk::process::name_t const proc_named = vistk::process::name_t("sink");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);

  pipeline->setup_pipeline();

  vistk::process::port_t const iport_name = vistk::process::port_t("color");
  vistk::process::port_t const oport_name = vistk::process::port_heartbeat;

  EXPECT_EXCEPTION(vistk::connection_after_setup_exception,
                   pipeline->connect(proc_named, oport_name,
                                     proc_nameu, iport_name),
                   "making a connection after setup");
}

void
test_setup_pipeline_map_input()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("sink");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("number");
  vistk::process::name_t const proc_named = vistk::process::name_t("sink");
  vistk::process::name_t const group_name = vistk::process::name_t("group");

  vistk::process_t const processu = create_process(proc_typeu, proc_nameu);
  vistk::process_t const processd = create_process(proc_typed, proc_named);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(processu);
  pipeline->add_process(processd);
  pipeline->add_group(group_name);

  vistk::process::port_t const port_nameu = vistk::process::port_t("number");
  vistk::process::port_t const port_named = vistk::process::port_t("sink");

  pipeline->connect(proc_nameu, port_nameu,
                    proc_named, port_named);

  pipeline->setup_pipeline();

  vistk::process::port_t const iport_name = vistk::process::port_t("color");

  EXPECT_EXCEPTION(vistk::connection_after_setup_exception,
                   pipeline->map_input_port(group_name, iport_name,
                                            proc_nameu, iport_name,
                                            vistk::process::port_flags_t()),
                   "mapping an input port after setup");
}

void
test_setup_pipeline_map_output()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process::name_t const proc_name = vistk::process::name_t("orphan");
  vistk::process::name_t const group_name = vistk::process::name_t("group");

  vistk::process_t const process = create_process(proc_type, proc_name);

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_process(process);
  pipeline->add_group(group_name);

  pipeline->setup_pipeline();

  vistk::process::port_t const port_name = vistk::process::port_heartbeat;

  EXPECT_EXCEPTION(vistk::connection_after_setup_exception,
                   pipeline->map_output_port(group_name, port_name,
                                             proc_name, port_name,
                                             vistk::process::port_flags_t()),
                   "mapping an output port after setup");
}

void
test_setup_pipeline()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typed = vistk::process_registry::type_t("multiplication");
  vistk::process_registry::type_t const proc_typet = vistk::process_registry::type_t("print_number");

  vistk::process::name_t const proc_nameu1 = vistk::process::name_t("upstream1");
  vistk::process::name_t const proc_nameu2 = vistk::process::name_t("upstream2");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  vistk::config_t configt = vistk::config::empty_config();

  vistk::config::key_t const output_key = vistk::config::key_t("output");
  vistk::config::value_t const output_path = vistk::config::value_t("test-pipeline-setup_pipeline-print_number.txt");

  configt->set_value(output_key, output_path);

  vistk::process_t const processu1 = create_process(proc_typeu, proc_nameu1);
  vistk::process_t const processu2 = create_process(proc_typeu, proc_nameu2);
  vistk::process_t const processd = create_process(proc_typed, proc_named);
  vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

  vistk::pipeline_t pipeline = create_pipeline();

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

vistk::process_t
create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name, vistk::config_t config)
{
  static bool const modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  config->set_value(vistk::process::config_name, vistk::config::value_t(name));

  return reg->create_process(type, config);
}

vistk::pipeline_t
create_pipeline()
{
  static vistk::config_t const config = vistk::config::empty_config();

  return boost::make_shared<vistk::pipeline>(config);
}
