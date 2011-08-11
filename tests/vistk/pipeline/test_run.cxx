/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <exception>
#include <fstream>
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

static void test_simple_pipeline();

void
run_test(std::string const& test_name)
{
  if (test_name == "simple_pipeline")
  {
    test_simple_pipeline();
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

static vistk::process_t create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name, vistk::config_t config = vistk::config::empty_config());
static vistk::pipeline_t create_pipeline();

void
test_simple_pipeline()
{
  vistk::process_registry::type_t const proc_typeu = vistk::process_registry::type_t("numbers");
  vistk::process_registry::type_t const proc_typet = vistk::process_registry::type_t("print_number");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  std::string const output_path = "test-run-simple_pipeline-print_number.txt";

  uint32_t const start_value = 10;
  uint32_t const end_value = 20;

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
    vistk::config::value_t const output_value = vistk::config::value_t("test-run-simple_pipeline-print_number.txt");

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

    vistk::schedule_registry::type_t const schedule_type = vistk::schedule_registry::type_t("thread_per_process");
    vistk::config_t const schedule_config = vistk::config::empty_config();

    vistk::schedule_t schedule = reg->create_schedule(schedule_type, schedule_config, pipeline);

    schedule->start();
    schedule->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    std::cerr << "Error: Could not open the output file" << std::endl;
  }

  std::string line;

  for (uint32_t i = start_value; i < end_value; ++i)
  {
    std::getline(fin, line);

    if (vistk::config::value_t(line) != boost::lexical_cast<vistk::config::value_t>(i))
    {
      std::cerr << "Error: Did not get expected value: "
                << "Expected: " << i << " "
                << "Received: " << line << std::endl;
    }
  }

  std::getline(fin, line);

  if (!line.empty())
  {
    std::cerr << "Error: Empty line missing" << std::endl;
  }

  if (!fin.eof())
  {
    std::cerr << "Error: Not at end of file" << std::endl;
  }
}

vistk::process_t
create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name, vistk::config_t config)
{
  static bool modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  config->set_value(vistk::process::config_name, vistk::config::value_t(name));

  return reg->create_process(type, config);
}

vistk::pipeline_t
create_pipeline()
{
  static vistk::config_t const config = vistk::config::empty_config();

  return vistk::pipeline_t(new vistk::pipeline(config));
}
