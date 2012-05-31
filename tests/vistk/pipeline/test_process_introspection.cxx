/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <exception>
#include <iostream>

#include <cstdlib>

static void test_process(vistk::process::type_t const& type);

int
main()
{
  try
  {
    vistk::load_known_modules();
  }
  catch (vistk::process_type_already_exists_exception& e)
  {
    TEST_ERROR("Duplicate process names: " << e.what());
  }
  catch (vistk::pipeline_exception& e)
  {
    TEST_ERROR("Failed to load modules: " << e.what());
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception when loading modules: " << e.what());

    return EXIT_FAILURE;
  }

  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::types_t const types = reg->types();

  BOOST_FOREACH (vistk::process::type_t const& type, types)
  {
    try
    {
      test_process(type);
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when testing "
                 "type \'" << type << "\': " << e.what());
    }
  }

  return EXIT_SUCCESS;
}

static void test_process_constraints(vistk::process_t const process);
static void test_process_configuration(vistk::process_t const process);
static void test_process_input_ports(vistk::process_t const process);
static void test_process_output_ports(vistk::process_t const process);

static void test_process_invalid_configuration(vistk::process_t const process);
static void test_process_invalid_input_port(vistk::process_t const process);
static void test_process_invalid_output_port(vistk::process_t const process);

void
test_process(vistk::process::type_t const& type)
{
  static vistk::process::name_t const expected_name = vistk::process::name_t("expected_name");

  vistk::process_registry_t const reg = vistk::process_registry::self();

  if (reg->description(type).empty())
  {
    TEST_ERROR("The description is empty");
  }

  vistk::process_t const process = reg->create_process(type, expected_name);

  if (!process)
  {
    TEST_ERROR("Received NULL process (" << type << ")");

    return;
  }

  if (process->name() != expected_name)
  {
    TEST_ERROR("Name (" << process->name() << ") "
               "does not match expected name: " << expected_name);
  }

  if (process->type() != type)
  {
    TEST_ERROR("The type does not match the registry type");
  }

  test_process_constraints(process);
  test_process_configuration(process);
  test_process_input_ports(process);
  test_process_output_ports(process);

  test_process_invalid_configuration(process);
  test_process_invalid_input_port(process);
  test_process_invalid_output_port(process);
}

void
test_process_constraints(vistk::process_t const process)
{
  vistk::process::constraints_t const consts = process->constraints();

  (void)consts;

  /// \todo Test for conflicting constraints.
}

void
test_process_configuration(vistk::process_t const process)
{
  vistk::config::keys_t const keys = process->available_config();

  BOOST_FOREACH (vistk::config::key_t const& key, keys)
  {
    try
    {
      process->config_info(key);
    }
    catch (vistk::unknown_configuration_value_exception& e)
    {
      TEST_ERROR("Failed to get a default for "
                 << process->type() << vistk::config::block_sep << key
                 << ": " << e.what());
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when querying for default "
                 "(" << process->type() << vistk::config::block_sep
                 << key << "): " << e.what());
    }
  }
}

void
test_process_input_ports(vistk::process_t const process)
{
  static vistk::config_t const config = vistk::config::empty_config();

  vistk::process::ports_t const ports = process->input_ports();

  BOOST_FOREACH (vistk::process::port_t const& port, ports)
  {
    vistk::process::port_info_t info;

    try
    {
      info = process->input_port_info(port);
    }
    catch (vistk::no_such_port_exception& e)
    {
      TEST_ERROR("Failed to get a info for input port "
                 << process->type() << "." << port << ": " << e.what());
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when querying for input port info "
                 "(" << process->type() << "." << port << "): " << e.what());
    }

    vistk::process::port_flags_t const& flags = info->flags;

    bool const is_const = (flags.find(vistk::process::flag_output_const) != flags.end());

    if (is_const)
    {
      TEST_ERROR("Const flag on input port "
                 "(" << process->type() << "." << port << ")");
    }

    vistk::process::port_type_t const& type = info->type;

    bool const is_data_dependent = (type == vistk::process::type_data_dependent);

    if (is_data_dependent)
    {
      TEST_ERROR("Data-dependent input port "
                 "(" << process->type() << "." << port << ")");
    }

    vistk::process::port_description_t const& description = info->description;

    if (description.empty())
    {
      TEST_ERROR("Description empty on input port "
                 "(" << process->type() << "." << port << ")");
    }

    vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

    process->connect_input_port(port, edge);

    EXPECT_EXCEPTION(vistk::port_reconnect_exception,
                     process->connect_input_port(port, edge),
                     "connecting to an input port a second time");
  }
}

void
test_process_output_ports(vistk::process_t const process)
{
  static vistk::config_t const config = vistk::config::empty_config();

  vistk::process::ports_t const ports = process->output_ports();

  BOOST_FOREACH (vistk::process::port_t const& port, ports)
  {
    vistk::process::port_info_t info;

    try
    {
      info = process->output_port_info(port);
    }
    catch (vistk::no_such_port_exception& e)
    {
      TEST_ERROR("Failed to get a info for output port "
                 << process->type() << "." << port << ": " << e.what());
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when querying for output port info "
                 "(" << process->type() << "." << port << "): " << e.what());
    }

    vistk::process::port_flags_t const& flags = info->flags;

    bool const is_mutable = (flags.find(vistk::process::flag_input_mutable) != flags.end());

    if (is_mutable)
    {
      TEST_ERROR("Mutable flag on output port "
                 "(" << process->type() << "." << port << ")");
    }

    bool const is_nodep = (flags.find(vistk::process::flag_input_nodep) != flags.end());

    if (is_nodep)
    {
      TEST_ERROR("No dependency flag on output port "
                 "(" << process->type() << "." << port << ")");
    }

    vistk::process::port_description_t const& description = info->description;

    if (description.empty())
    {
      TEST_ERROR("Description empty on output port "
                 "(" << process->type() << "." << port << ")");
    }

    vistk::edge_t edge1 = boost::make_shared<vistk::edge>(config);
    vistk::edge_t edge2 = boost::make_shared<vistk::edge>(config);

    process->connect_output_port(port, edge1);
    process->connect_output_port(port, edge2);
  }
}

void
test_process_invalid_configuration(vistk::process_t const process)
{
  vistk::config::key_t const non_existent_config = vistk::config::key_t("does_not_exist");

  EXPECT_EXCEPTION(vistk::unknown_configuration_value_exception,
                   process->config_info(non_existent_config),
                   "requesting the information for a non-existent config");
}

void
test_process_invalid_input_port(vistk::process_t const process)
{
  static vistk::process::port_t const non_existent_port = vistk::process::port_t("does_not_exist");
  static vistk::config_t const config = vistk::config::empty_config();

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->input_port_info(non_existent_port),
                   "requesting the info for a non-existent input port");

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->connect_input_port(non_existent_port, edge),
                   "requesting a connection to a non-existent input port");
}

void
test_process_invalid_output_port(vistk::process_t const process)
{
  static vistk::process::port_t const non_existent_port = vistk::process::port_t("does_not_exist");
  static vistk::config_t const config = vistk::config::empty_config();

  // Output ports.
  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->output_port_info(non_existent_port),
                   "requesting the info for a non-existent output port");

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->connect_output_port(non_existent_port, edge),
                   "requesting a connection to a non-existent output port");
}
