/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/dtor_registry.h>
#include <vistk/pipeline/dtor_registry_exception.h>

#include <exception>
#include <iostream>
#include <string>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return 1;
  }

  return 0;
}

static void test_null_dtor();
static void test_call_dtor();
static void test_module_marking();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_dtor")
  {
    test_null_dtor();
  }
  else if (test_name == "call_dtor")
  {
    test_call_dtor();
  }
  else if (test_name == "module_marking")
  {
    test_module_marking();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_null_dtor()
{
  vistk::dtor_registry_t reg = vistk::dtor_registry::self();

  vistk::config_t config;

  EXPECT_EXCEPTION(vistk::null_dtor_exception,
                   reg->register_dtor(vistk::dtor_t()),
                   "passing a NULL dtor to the registry");
}

static void dtor_call();

void
test_call_dtor()
{
  vistk::dtor_registry_t reg = vistk::dtor_registry::self();

  reg->register_dtor(&dtor_call);
}

void
dtor_call()
{
  std::cout << "PASS: Dtor called" << std::endl;
}

void
test_module_marking()
{
  vistk::dtor_registry_t reg = vistk::dtor_registry::self();

  vistk::dtor_registry::module_t const module = vistk::dtor_registry::module_t("module");

  if (reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "already marked as loaded");
  }

  reg->mark_module_as_loaded(module);

  if (!reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "not marked as loaded");
  }
}
