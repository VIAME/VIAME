/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>
#include <vistk/pipeline/process.h>

#include <exception>
#include <iostream>
#include <string>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Error: Expected one argument" << std::endl;

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: " << e.what() << std::endl;

    return 1;
  }

  return 0;
}

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

void
run_test(std::string const& test_name)
{
  if (test_name == "null_process")
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
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

void
test_null_process()
{
  vistk::process_t const process;

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  bool got_exception = false;

  try
  {
    pipeline->add_process(process);
  }
  catch (vistk::null_process_addition_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when adding a NULL process to the pipeline" << std::endl;
  }
}

void
test_add_process()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::process_t const process = reg->create_process(proc_type, config);

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_process(process);
}

void
test_add_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_group(proc_name);
}

void
test_duplicate_process_process()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t proc_config = vistk::config::empty_config();

  proc_config->set_value(vistk::process::config_name, proc_name);

  vistk::process_t const process = reg->create_process(proc_type, proc_config);
  vistk::process_t const dup_process = reg->create_process(proc_type, proc_config);

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_process(process);

  bool got_exception = false;

  try
  {
    pipeline->add_process(dup_process);
  }
  catch (vistk::duplicate_process_name_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when adding a duplicate process to the pipeline" << std::endl;
  }
}

void
test_duplicate_process_group()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t proc_config = vistk::config::empty_config();

  proc_config->set_value(vistk::process::config_name, proc_name);

  vistk::process_t const dup_process = reg->create_process(proc_type, proc_config);

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_group(proc_name);

  bool got_exception = false;

  try
  {
    pipeline->add_process(dup_process);
  }
  catch (vistk::duplicate_process_name_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when adding a duplicate process to the pipeline" << std::endl;
  }
}

void
test_duplicate_group_process()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("numbers");

  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t proc_config = vistk::config::empty_config();

  proc_config->set_value(vistk::process::config_name, proc_name);

  vistk::process_t const process = reg->create_process(proc_type, proc_config);

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_process(process);

  bool got_exception = false;

  try
  {
    pipeline->add_group(proc_name);
  }
  catch (vistk::duplicate_process_name_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when adding a duplicate group to the pipeline" << std::endl;
  }
}

void
test_duplicate_group_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_group(proc_name);

  bool got_exception = false;

  try
  {
    pipeline->add_group(proc_name);
  }
  catch (vistk::duplicate_process_name_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when adding a duplicate group to the pipeline" << std::endl;
  }
}

void
test_map_input_no_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  bool got_exception = false;

  try
  {
    pipeline->map_input_port(proc_name, vistk::process::port_t(),
                             vistk::process::name_t(), vistk::process::port_t(),
                             vistk::process::port_flags_t());
  }
  catch (vistk::no_such_group_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when mapping an input on an non-existent group" << std::endl;
  }
}

void
test_map_output_no_group()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  bool got_exception = false;

  try
  {
    pipeline->map_output_port(proc_name, vistk::process::port_t(),
                              vistk::process::name_t(), vistk::process::port_t(),
                              vistk::process::port_flags_t());
  }
  catch (vistk::no_such_group_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when mapping an output on an non-existent group" << std::endl;
  }
}

void
test_map_input_no_process()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t pipeline = vistk::pipeline_t(new vistk::pipeline(config));

  pipeline->add_group(proc_name);

  bool got_exception = false;

  try
  {
    pipeline->map_input_port(proc_name, vistk::process::port_t(),
                             vistk::process::name_t(), vistk::process::port_t(),
                             vistk::process::port_flags_t());
  }
  catch (vistk::no_such_process_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when mapping an input on an non-existent group" << std::endl;
  }
}
