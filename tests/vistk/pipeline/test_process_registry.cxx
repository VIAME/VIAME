/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process_cluster.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>

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
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_get_twice();
static void test_null_config();
static void test_load_processes();
static void test_null_ctor();
static void test_duplicate_types();
static void test_unknown_types();
static void test_module_marking();
static void test_register_cluster();

void
run_test(std::string const& test_name)
{
  if (test_name == "get_twice")
  {
    test_get_twice();
  }
  else if (test_name == "null_config")
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
  else if (test_name == "module_marking")
  {
    test_module_marking();
  }
  else if (test_name == "register_cluster")
  {
    test_register_cluster();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_get_twice()
{
  vistk::process_registry_t reg1 = vistk::process_registry::self();
  vistk::process_registry_t reg2 = vistk::process_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

void
test_null_config()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::config_t config;

  EXPECT_EXCEPTION(vistk::null_process_registry_config_exception,
                   reg->create_process(vistk::process::type_t(), vistk::process::name_t(), config),
                   "requesting a NULL config to a process");
}

void
test_load_processes()
{
  vistk::load_known_modules();

  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process::types_t const types = reg->types();

  BOOST_FOREACH (vistk::process::type_t const& type, types)
  {
    vistk::process_t process;

    try
    {
      process = reg->create_process(type, vistk::process::name_t());
    }
    catch (vistk::no_such_process_type_exception const& e)
    {
      TEST_ERROR("Failed to create process: " << e.what());

      continue;
    }
    catch (std::exception const& e)
    {
      TEST_ERROR("Unexpected exception when creating process: " << e.what());

      continue;
    }

    if (!process)
    {
      TEST_ERROR("Received NULL process (" << type << ")");

      continue;
    }

    if (reg->description(type).empty())
    {
      TEST_ERROR("The description for "
                 << type << " is empty");
    }
  }
}

void
test_null_ctor()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  EXPECT_EXCEPTION(vistk::null_process_ctor_exception,
                   reg->register_process(vistk::process::type_t(), vistk::process_registry::description_t(), vistk::process_ctor_t()),
                   "requesting an non-existent process type");
}

static vistk::process_t null_process(vistk::config_t const& config);

void
test_duplicate_types()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process::type_t const non_existent_process = vistk::process::type_t("no_such_process");

  reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process);

  EXPECT_EXCEPTION(vistk::process_type_already_exists_exception,
                   reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process),
                   "requesting an non-existent process type");
}

void
test_unknown_types()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process::type_t const non_existent_process = vistk::process::type_t("no_such_process");

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->create_process(non_existent_process, vistk::process::name_t()),
                   "requesting an non-existent process type");

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->description(non_existent_process),
                   "requesting an non-existent process type");
}

void
test_module_marking()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process_registry::module_t const module = vistk::process_registry::module_t("module");

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

void
test_register_cluster()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const cluster_type = vistk::process::type_t("orphan_cluster");
  vistk::config_t const config = vistk::config::empty_config();

  vistk::process_t const cluster_from_reg = reg->create_process(cluster_type, vistk::process::name_t(), config);

  vistk::process_cluster_t const cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(cluster_from_reg);

  if (!cluster)
  {
    TEST_ERROR("Failed to turn a process back into a cluster");
  }

  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::process_t const not_a_cluster_from_reg = reg->create_process(type, vistk::process::name_t(), config);

  vistk::process_cluster_t const not_a_cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(not_a_cluster_from_reg);

  if (not_a_cluster)
  {
    TEST_ERROR("Turned a non-cluster into a cluster");
  }
}

vistk::process_t
null_process(vistk::config_t const& /*config*/)
{
  return vistk::process_t();
}
