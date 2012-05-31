/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>

#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include <cstddef>
#include <cstdlib>

static std::string const test_sep = "-";

static void run_test(std::string const& test_name, vistk::schedule_registry::type_t const& schedule_type);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const full_test_name = argv[1];

  size_t sep_pos = full_test_name.find(test_sep);

  if (sep_pos == std::string::npos)
  {
    TEST_ERROR("Unexpected test name format: " << full_test_name);

    return EXIT_FAILURE;
  }

  std::string const test_name = full_test_name.substr(0, sep_pos);
  std::string const schedule_type = full_test_name.substr(sep_pos + test_sep.length());

  try
  {
    run_test(test_name, schedule_type);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_simple_pipeline(vistk::schedule_registry::type_t const& schedule_type);
static void test_multiplier_pipeline(vistk::schedule_registry::type_t const& schedule_type);

void
run_test(std::string const& test_name, vistk::schedule_registry::type_t const& schedule_type)
{
  if (test_name == "simple_pipeline")
  {
    test_simple_pipeline(schedule_type);
  }
  else if (test_name == "multiplier_pipeline")
  {
    test_multiplier_pipeline(schedule_type);
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

static vistk::process_t create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config = vistk::config::empty_config());
static vistk::pipeline_t create_pipeline();

void
test_simple_pipeline(vistk::schedule_registry::type_t const& schedule_type)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typet = vistk::process::type_t("print_number");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  std::string const output_path = "test-run-simple_pipeline-" + schedule_type + "-print_number.txt";

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    vistk::config_t configu = vistk::config::empty_config();

    vistk::config::key_t const start_key = vistk::config::key_t("start");
    vistk::config::value_t const start_num = boost::lexical_cast<vistk::config::value_t>(start_value);
    vistk::config::key_t const end_key = vistk::config::key_t("end");
    vistk::config::value_t const end_num = boost::lexical_cast<vistk::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    vistk::config_t configt = vistk::config::empty_config();

    vistk::config::key_t const output_key = vistk::config::key_t("output");
    vistk::config::value_t const output_value = vistk::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    vistk::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

    vistk::pipeline_t pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(processt);

    vistk::process::port_t const port_nameu = vistk::process::port_t("number");
    vistk::process::port_t const port_namet = vistk::process::port_t("number");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

    vistk::schedule_t schedule = reg->create_schedule(schedule_type, pipeline);

    schedule->start();
    schedule->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  for (int32_t i = start_value; i < end_value; ++i)
  {
    std::getline(fin, line);

    if (vistk::config::value_t(line) != boost::lexical_cast<vistk::config::value_t>(i))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i << " "
                 "Received: " << line);
    }
  }

  std::getline(fin, line);

  if (!line.empty())
  {
    TEST_ERROR("Empty line missing");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

void
test_multiplier_pipeline(vistk::schedule_registry::type_t const& schedule_type)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_types = vistk::process::type_t("source");
  vistk::process::type_t const proc_typed = vistk::process::type_t("multiplication");
  vistk::process::type_t const proc_typet = vistk::process::type_t("print_number");

  vistk::process::name_t const proc_nameu1 = vistk::process::name_t("upstream1");
  vistk::process::name_t const proc_nameu2 = vistk::process::name_t("upstream2");
  vistk::process::name_t const proc_names = vistk::process::name_t("source");
  vistk::process::name_t const proc_named = vistk::process::name_t("downstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  std::string const output_path = "test-run-multiplier_pipeline-" + schedule_type + "-print_number.txt";

  int32_t const start_value1 = 10;
  int32_t const end_value1 = 20;

  int32_t const start_value2 = 10;
  int32_t const end_value2= 30;

  {
    vistk::config_t configu1 = vistk::config::empty_config();

    vistk::config::key_t const start_key = vistk::config::key_t("start");
    vistk::config::key_t const end_key = vistk::config::key_t("end");

    vistk::config::value_t const start_num1 = boost::lexical_cast<vistk::config::value_t>(start_value1);
    vistk::config::value_t const end_num1 = boost::lexical_cast<vistk::config::value_t>(end_value1);

    configu1->set_value(start_key, start_num1);
    configu1->set_value(end_key, end_num1);

    vistk::config_t configu2 = vistk::config::empty_config();

    vistk::config::value_t const start_num2 = boost::lexical_cast<vistk::config::value_t>(start_value2);
    vistk::config::value_t const end_num2 = boost::lexical_cast<vistk::config::value_t>(end_value2);

    configu2->set_value(start_key, start_num2);
    configu2->set_value(end_key, end_num2);

    vistk::config_t configt = vistk::config::empty_config();

    vistk::config::key_t const output_key = vistk::config::key_t("output");
    vistk::config::value_t const output_value = vistk::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    vistk::process_t const processu1 = create_process(proc_typeu, proc_nameu1, configu1);
    vistk::process_t const processu2 = create_process(proc_typeu, proc_nameu2, configu2);
    vistk::process_t const processs = create_process(proc_types, proc_names);
    vistk::process_t const processd = create_process(proc_typed, proc_named);
    vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

    vistk::pipeline_t pipeline = create_pipeline();

    pipeline->add_process(processu1);
    pipeline->add_process(processu2);
    pipeline->add_process(processs);
    pipeline->add_process(processd);
    pipeline->add_process(processt);

    vistk::process::port_t const port_nameu = vistk::process::port_t("number");
    vistk::process::port_t const port_namesi1 = vistk::process::port_t("src/factor1");
    vistk::process::port_t const port_namesi2 = vistk::process::port_t("src/factor2");
    vistk::process::port_t const port_nameso1 = vistk::process::port_t("out/factor1");
    vistk::process::port_t const port_nameso2 = vistk::process::port_t("out/factor2");
    vistk::process::port_t const port_named1 = vistk::process::port_t("factor1");
    vistk::process::port_t const port_named2 = vistk::process::port_t("factor2");
    vistk::process::port_t const port_namedo = vistk::process::port_t("product");
    vistk::process::port_t const port_namet = vistk::process::port_t("number");

    pipeline->connect(proc_nameu1, port_nameu,
                      proc_names, port_namesi1);
    pipeline->connect(proc_nameu2, port_nameu,
                      proc_names, port_namesi2);
    pipeline->connect(proc_names, port_nameso1,
                      proc_named, port_named1);
    pipeline->connect(proc_names, port_nameso2,
                      proc_named, port_named2);
    pipeline->connect(proc_named, port_namedo,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

    vistk::schedule_t schedule = reg->create_schedule(schedule_type, pipeline);

    schedule->start();
    schedule->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  for (int32_t i = start_value1, j = start_value2;
       (i < end_value1) && (j < end_value2); ++i, ++j)
  {
    std::getline(fin, line);

    if (vistk::config::value_t(line) != boost::lexical_cast<vistk::config::value_t>(i * j))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i * j << " "
                 "Received: " << line);
    }
  }

  std::getline(fin, line);

  if (!line.empty())
  {
    TEST_ERROR("Empty line missing");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

vistk::process_t
create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config)
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
  return boost::make_shared<vistk::pipeline>();
}
