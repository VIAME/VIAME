/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
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

#define TEST_ARGS ()

DECLARE_TEST(get_twice);
DECLARE_TEST(null_config);
DECLARE_TEST(load_processes);
DECLARE_TEST(null_ctor);
DECLARE_TEST(duplicate_types);
DECLARE_TEST(unknown_types);
DECLARE_TEST(module_marking);
DECLARE_TEST(register_cluster);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, get_twice);
  ADD_TEST(tests, null_config);
  ADD_TEST(tests, load_processes);
  ADD_TEST(tests, null_ctor);
  ADD_TEST(tests, duplicate_types);
  ADD_TEST(tests, unknown_types);
  ADD_TEST(tests, module_marking);
  ADD_TEST(tests, register_cluster);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(get_twice)
{
  vistk::process_registry_t const reg1 = vistk::process_registry::self();
  vistk::process_registry_t const reg2 = vistk::process_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

IMPLEMENT_TEST(null_config)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::config_t const config;

  EXPECT_EXCEPTION(vistk::null_process_registry_config_exception,
                   reg->create_process(vistk::process::type_t(), vistk::process::name_t(), config),
                   "requesting a NULL config to a process");
}

IMPLEMENT_TEST(load_processes)
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();

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

IMPLEMENT_TEST(null_ctor)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  EXPECT_EXCEPTION(vistk::null_process_ctor_exception,
                   reg->register_process(vistk::process::type_t(), vistk::process_registry::description_t(), vistk::process_ctor_t()),
                   "requesting an non-existent process type");
}

static vistk::process_t null_process(vistk::config_t const& config);

IMPLEMENT_TEST(duplicate_types)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const non_existent_process = vistk::process::type_t("no_such_process");

  reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process);

  EXPECT_EXCEPTION(vistk::process_type_already_exists_exception,
                   reg->register_process(non_existent_process, vistk::process_registry::description_t(), null_process),
                   "requesting an non-existent process type");
}

IMPLEMENT_TEST(unknown_types)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const non_existent_process = vistk::process::type_t("no_such_process");

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->create_process(non_existent_process, vistk::process::name_t()),
                   "requesting an non-existent process type");

  EXPECT_EXCEPTION(vistk::no_such_process_type_exception,
                   reg->description(non_existent_process),
                   "requesting an non-existent process type");
}

IMPLEMENT_TEST(module_marking)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

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

IMPLEMENT_TEST(register_cluster)
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
