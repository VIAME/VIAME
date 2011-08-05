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
static void test_map_output_no_process();
static void test_map_input();
static void test_map_output();
static void test_connect_no_upstream();
static void test_connect_no_downstream();
static void test_connect_type_mismatch();
static void test_connect_flag_mismatch();
static void test_connect();
static void test_connect_input_map();
static void test_connect_output_map();

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
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

#define EXPECT_EXCEPTION(exc, code, action)        \
  do                                               \
  {                                                \
    bool got_exception = false;                    \
                                                   \
    try                                            \
    {                                              \
      code;                                        \
    }                                              \
    catch (exc& e)                                 \
    {                                              \
      got_exception = true;                        \
    }                                              \
    catch (std::exception& e)                      \
    {                                              \
      std::cerr << "Error: Unexpected exception: " \
                << e.what() << std::endl;          \
                                                   \
      got_exception = true;                        \
    }                                              \
                                                   \
    if (!got_exception)                            \
    {                                              \
      std::cerr << "Error: Did not get "           \
                << "expected exception when "      \
                << action << std::endl;            \
    }                                              \
  } while (false)


static vistk::process_t create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name);
static vistk::pipeline_t create_pipeline();

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

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->map_input_port(proc_name, vistk::process::port_t(),
                                            vistk::process::name_t(), vistk::process::port_t(),
                                            vistk::process::port_flags_t()),
                   "mapping an input on an non-existent group");
}

void
test_map_output_no_process()
{
  vistk::config::value_t const proc_name = vistk::process::name_t("name");

  vistk::pipeline_t pipeline = create_pipeline();

  pipeline->add_group(proc_name);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   pipeline->map_output_port(proc_name, vistk::process::port_t(),
                                             vistk::process::name_t(), vistk::process::port_t(),
                                             vistk::process::port_flags_t()),
                   "mapping an output on a non-existent group");
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
test_connect_type_mismatch()
{
  /// \todo Need processes with type mismatches first.
  std::cerr << "Error: Not implemented" << std::endl;
}

void
test_connect_flag_mismatch()
{
  /// \todo Need processes with flag mismatches first.
  std::cerr << "Error: Not implemented" << std::endl;
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
  std::cerr << "Error: Not implemented" << std::endl;
}

void
test_connect_output_map()
{
  std::cerr << "Error: Not implemented" << std::endl;
}

vistk::process_t
create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name)
{
  static bool modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  vistk::config_t config = vistk::config::empty_config();

  config->set_value(vistk::process::config_name, vistk::config::value_t(name));

  return reg->create_process(type, config);
}

vistk::pipeline_t
create_pipeline()
{
  static vistk::config_t const config = vistk::config::empty_config();

  return vistk::pipeline_t(new vistk::pipeline(config));
}
