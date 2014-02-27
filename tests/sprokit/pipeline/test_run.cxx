/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>

#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include <fstream>

#define TEST_ARGS (sprokit::scheduler_registry::type_t const& scheduler_type)

DECLARE_TEST_MAP();

static std::string const test_sep = "-";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const full_testname = argv[1];

  size_t const sep_pos = full_testname.find(test_sep);

  if (sep_pos == testname_t::npos)
  {
    TEST_ERROR("Unexpected test name format: " << full_testname);

    return EXIT_FAILURE;
  }

  testname_t const testname = full_testname.substr(0, sep_pos);
  sprokit::scheduler_registry::type_t const scheduler_type = full_testname.substr(sep_pos + test_sep.length());

  RUN_TEST(testname, scheduler_type);
}

static sprokit::process_t create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name, sprokit::config_t config = sprokit::config::empty_config());
static sprokit::pipeline_t create_pipeline();

IMPLEMENT_TEST(simple_pipeline)
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t("numbers");
  sprokit::process::type_t const proc_typet = sprokit::process::type_t("print_number");

  sprokit::process::name_t const proc_nameu = sprokit::process::name_t("upstream");
  sprokit::process::name_t const proc_namet = sprokit::process::name_t("terminal");

  std::string const output_path = "test-run-simple_pipeline-" + scheduler_type + "-print_number.txt";

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    sprokit::config_t const configu = sprokit::config::empty_config();

    sprokit::config::key_t const start_key = sprokit::config::key_t("start");
    sprokit::config::value_t const start_num = boost::lexical_cast<sprokit::config::value_t>(start_value);
    sprokit::config::key_t const end_key = sprokit::config::key_t("end");
    sprokit::config::value_t const end_num = boost::lexical_cast<sprokit::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    sprokit::config_t const configt = sprokit::config::empty_config();

    sprokit::config::key_t const output_key = sprokit::config::key_t("output");
    sprokit::config::value_t const output_value = sprokit::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    sprokit::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    sprokit::process_t const processt = create_process(proc_typet, proc_namet, configt);

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(processt);

    sprokit::process::port_t const port_nameu = sprokit::process::port_t("number");
    sprokit::process::port_t const port_namet = sprokit::process::port_t("number");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

    sprokit::scheduler_t const scheduler = reg->create_scheduler(scheduler_type, pipeline);

    scheduler->start();
    scheduler->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  for (int32_t i = start_value; i < end_value; ++i)
  {
    if (!std::getline(fin, line))
    {
      TEST_ERROR("Failed to read a line from the file");
    }

    if (sprokit::config::value_t(line) != boost::lexical_cast<sprokit::config::value_t>(i))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i << " "
                 "Received: " << line);
    }
  }

  if (std::getline(fin, line))
  {
    TEST_ERROR("More results than expected in the file");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

IMPLEMENT_TEST(pysimple_pipeline)
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t("numbers");
  sprokit::process::type_t const proc_typet = sprokit::process::type_t("pyprint_number");

  sprokit::process::name_t const proc_nameu = sprokit::process::name_t("upstream");
  sprokit::process::name_t const proc_namet = sprokit::process::name_t("terminal");

  std::string const output_path = "test-run-pysimple_pipeline-" + scheduler_type + "-print_number.txt";

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    sprokit::config_t const configu = sprokit::config::empty_config();

    sprokit::config::key_t const start_key = sprokit::config::key_t("start");
    sprokit::config::value_t const start_num = boost::lexical_cast<sprokit::config::value_t>(start_value);
    sprokit::config::key_t const end_key = sprokit::config::key_t("end");
    sprokit::config::value_t const end_num = boost::lexical_cast<sprokit::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    sprokit::config_t const configt = sprokit::config::empty_config();

    sprokit::config::key_t const output_key = sprokit::config::key_t("output");
    sprokit::config::value_t const output_value = sprokit::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    sprokit::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    sprokit::process_t const processt = create_process(proc_typet, proc_namet, configt);

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(processt);

    sprokit::process::port_t const port_nameu = sprokit::process::port_t("number");
    sprokit::process::port_t const port_namet = sprokit::process::port_t("input");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

    sprokit::scheduler_t const scheduler = reg->create_scheduler(scheduler_type, pipeline);

    scheduler->start();
    scheduler->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  for (int32_t i = start_value; i < end_value; ++i)
  {
    if (!std::getline(fin, line))
    {
      TEST_ERROR("Failed to read a line from the file");
    }

    if (sprokit::config::value_t(line) != boost::lexical_cast<sprokit::config::value_t>(i))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i << " "
                 "Received: " << line);
    }
  }

  if (std::getline(fin, line))
  {
    TEST_ERROR("More results than expected in the file");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

IMPLEMENT_TEST(multiplier_pipeline)
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t("numbers");
  sprokit::process::type_t const proc_typed = sprokit::process::type_t("multiplication");
  sprokit::process::type_t const proc_typet = sprokit::process::type_t("print_number");

  sprokit::process::name_t const proc_nameu1 = sprokit::process::name_t("upstream1");
  sprokit::process::name_t const proc_nameu2 = sprokit::process::name_t("upstream2");
  sprokit::process::name_t const proc_named = sprokit::process::name_t("downstream");
  sprokit::process::name_t const proc_namet = sprokit::process::name_t("terminal");

  std::string const output_path = "test-run-multiplier_pipeline-" + scheduler_type + "-print_number.txt";

  int32_t const start_value1 = 10;
  int32_t const end_value1 = 20;

  int32_t const start_value2 = 10;
  int32_t const end_value2= 30;

  {
    sprokit::config_t const configu1 = sprokit::config::empty_config();

    sprokit::config::key_t const start_key = sprokit::config::key_t("start");
    sprokit::config::key_t const end_key = sprokit::config::key_t("end");

    sprokit::config::value_t const start_num1 = boost::lexical_cast<sprokit::config::value_t>(start_value1);
    sprokit::config::value_t const end_num1 = boost::lexical_cast<sprokit::config::value_t>(end_value1);

    configu1->set_value(start_key, start_num1);
    configu1->set_value(end_key, end_num1);

    sprokit::config_t const configu2 = sprokit::config::empty_config();

    sprokit::config::value_t const start_num2 = boost::lexical_cast<sprokit::config::value_t>(start_value2);
    sprokit::config::value_t const end_num2 = boost::lexical_cast<sprokit::config::value_t>(end_value2);

    configu2->set_value(start_key, start_num2);
    configu2->set_value(end_key, end_num2);

    sprokit::config_t const configt = sprokit::config::empty_config();

    sprokit::config::key_t const output_key = sprokit::config::key_t("output");
    sprokit::config::value_t const output_value = sprokit::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    sprokit::process_t const processu1 = create_process(proc_typeu, proc_nameu1, configu1);
    sprokit::process_t const processu2 = create_process(proc_typeu, proc_nameu2, configu2);
    sprokit::process_t const processd = create_process(proc_typed, proc_named);
    sprokit::process_t const processt = create_process(proc_typet, proc_namet, configt);

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu1);
    pipeline->add_process(processu2);
    pipeline->add_process(processd);
    pipeline->add_process(processt);

    sprokit::process::port_t const port_nameu = sprokit::process::port_t("number");
    sprokit::process::port_t const port_named1 = sprokit::process::port_t("factor1");
    sprokit::process::port_t const port_named2 = sprokit::process::port_t("factor2");
    sprokit::process::port_t const port_namedo = sprokit::process::port_t("product");
    sprokit::process::port_t const port_namet = sprokit::process::port_t("number");

    pipeline->connect(proc_nameu1, port_nameu,
                      proc_named, port_named1);
    pipeline->connect(proc_nameu2, port_nameu,
                      proc_named, port_named2);
    pipeline->connect(proc_named, port_namedo,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

    sprokit::scheduler_t const scheduler = reg->create_scheduler(scheduler_type, pipeline);

    scheduler->start();
    scheduler->wait();
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
    if (!std::getline(fin, line))
    {
      TEST_ERROR("Failed to read a line from the file");
    }

    if (sprokit::config::value_t(line) != boost::lexical_cast<sprokit::config::value_t>(i * j))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i * j << " "
                 "Received: " << line);
    }
  }

  if (std::getline(fin, line))
  {
    TEST_ERROR("More results than expected in the file");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

IMPLEMENT_TEST(multiplier_cluster_pipeline)
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t("numbers");
  sprokit::process::type_t const proc_typed = sprokit::process::type_t("multiplier_cluster");
  sprokit::process::type_t const proc_typet = sprokit::process::type_t("print_number");

  sprokit::process::name_t const proc_nameu = sprokit::process::name_t("upstream");
  sprokit::process::name_t const proc_named = sprokit::process::name_t("downstream");
  sprokit::process::name_t const proc_namet = sprokit::process::name_t("terminal");

  std::string const output_path = "test-run-multiplier_cluster_pipeline-" + scheduler_type + "-print_number.txt";

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  int32_t const factor_value = 20;

  {
    sprokit::config_t const configu = sprokit::config::empty_config();

    sprokit::config::key_t const start_key = sprokit::config::key_t("start");
    sprokit::config::key_t const end_key = sprokit::config::key_t("end");

    sprokit::config::value_t const start_num = boost::lexical_cast<sprokit::config::value_t>(start_value);
    sprokit::config::value_t const end_num = boost::lexical_cast<sprokit::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    sprokit::config_t const configd = sprokit::config::empty_config();

    sprokit::config::key_t const factor_key = sprokit::config::key_t("factor");

    sprokit::config::value_t const factor = boost::lexical_cast<sprokit::config::value_t>(factor_value);

    configd->set_value(factor_key, factor);

    sprokit::config_t const configt = sprokit::config::empty_config();

    sprokit::config::key_t const output_key = sprokit::config::key_t("output");
    sprokit::config::value_t const output_value = sprokit::config::value_t(output_path);

    configt->set_value(output_key, output_value);

    sprokit::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    sprokit::process_t const processd = create_process(proc_typed, proc_named, configd);
    sprokit::process_t const processt = create_process(proc_typet, proc_namet, configt);

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(processd);
    pipeline->add_process(processt);

    sprokit::process::port_t const port_nameu = sprokit::process::port_t("number");
    sprokit::process::port_t const port_namedi = sprokit::process::port_t("factor");
    sprokit::process::port_t const port_namedo = sprokit::process::port_t("product");
    sprokit::process::port_t const port_namet = sprokit::process::port_t("number");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_named, port_namedi);
    pipeline->connect(proc_named, port_namedo,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

    sprokit::scheduler_t const scheduler = reg->create_scheduler(scheduler_type, pipeline);

    scheduler->start();
    scheduler->wait();
  }

  std::ifstream fin(output_path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  for (int32_t i = start_value; i < end_value; ++i)
  {
    if (!std::getline(fin, line))
    {
      TEST_ERROR("Failed to read a line from the file");
    }

    if (sprokit::config::value_t(line) != boost::lexical_cast<sprokit::config::value_t>(i * factor_value))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i * factor_value << " "
                 "Received: " << line);
    }
  }

  if (std::getline(fin, line))
  {
    TEST_ERROR("More results than expected in the file");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

sprokit::process_t
create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name, sprokit::config_t config)
{
  static bool const modules_loaded = (sprokit::load_known_modules(), true);
  static sprokit::process_registry_t const reg = sprokit::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name, config);
}

sprokit::pipeline_t
create_pipeline()
{
  return boost::make_shared<sprokit::pipeline>();
}
