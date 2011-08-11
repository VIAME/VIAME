/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>

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

static void test_null_config();
static void test_load_processes();
static void test_null_ctor();
static void test_duplicate_types();
static void test_unknown_types();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "load_processes")
  {
    test_load_processes();
  }
  else if (test_name == "null_ctor")
  {
    test_null_ctor();
  }
  else if (test_name == "duplicate_types")
  {
    test_duplicate_types();
  }
  else if (test_name == "unknown_types")
  {
    test_unknown_types();
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

void
test_null_config()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::config_t config;

  EXPECT_EXCEPTION(vistk::null_process_registry_config_exception,
                   reg->create_process(vistk::process_registry::type_t(), config),
                   "requesting a NULL config to a process");
}

void
test_load_processes()
{
  vistk::load_known_modules();

  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process_registry::types_t const types = reg->types();

  vistk::config_t config = vistk::config::empty_config();

  BOOST_FOREACH (vistk::process_registry::type_t const& type, types)
  {
    vistk::process_t process;

    try
    {
      process = reg->create_process(type, config);
    }
    catch (vistk::no_such_process_type_exception& e)
    {
      std::cerr << "Error: Failed to create process: " << e.what() << std::endl;

      continue;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when creating process: " << e.what() << std::endl;

      continue;
    }

    if (!process)
    {
      std::cerr << "Error: Received NULL process (" << type << ")" << std::endl;

      continue;
    }

    if (reg->description(type).empty())
    {
      std::cerr << "Error: The description for "
               << type << " is empty" << std::endl;
    }
  }
}

void
test_null_ctor()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  EXPECT_EXCEPTION(vistk::null_process_ctor_exception,
                   reg->register_process(vistk::process_registry::type_t(), vistk::process_registry::description_t(), vistk::process_ctor_t()),
                   "requesting an non-existent process type");
}

static vistk::process_t null_process(vistk::config_t const& config);

void
test_duplicate_types()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process_registry::type_t const non_existent_process = vistk::process_registry::type_t("no_such_process");

  reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process);

  EXPECT_EXCEPTION(vistk::process_type_already_exists_exception,
                   reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process),
                   "requesting an non-existent process type");
}

void
test_unknown_types()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process_registry::type_t const non_existent_process = vistk::process_registry::type_t("no_such_process");

  vistk::config_t config = vistk::config::empty_config();

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->create_process(non_existent_process, config),
                   "requesting an non-existent process type");

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->description(non_existent_process),
                   "requesting an non-existent process type");
}

vistk::process_t
null_process(vistk::config_t const& /*config*/)
{
  return vistk::process_t();
}
