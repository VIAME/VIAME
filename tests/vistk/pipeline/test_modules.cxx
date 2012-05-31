/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

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

static void test_load();
static void test_multiple_load();
static void test_envvar();

void
run_test(std::string const& test_name)
{
  if (test_name == "load")
  {
    test_load();
  }
  else if (test_name == "multiple_load")
  {
    test_multiple_load();
  }
  else if (test_name == "envvar")
  {
    test_envvar();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_load()
{
  vistk::load_known_modules();
}

void
test_multiple_load()
{
  vistk::load_known_modules();
  vistk::load_known_modules();
}

void
test_envvar()
{
  vistk::load_known_modules();

  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("test");

  reg->create_process(proc_type);
}
