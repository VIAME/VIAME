/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>

#include <exception>
#include <iostream>

static void test_process(vistk::process_registry::type_t const& type);

int
main()
{
  try
  {
    vistk::load_known_modules();
  }
  catch (vistk::process_type_already_exists_exception& e)
  {
    std::cerr << "Error: Duplicate process names: " << e.what() << std::endl;
  }
  catch (vistk::pipeline_exception& e)
  {
    std::cerr << "Error: Failed to load modules: " << e.what() << std::endl;
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception when loading modules: " << e.what() << std::endl;

    return 1;
  }

  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process_registry::types_t const types = reg->types();

  BOOST_FOREACH (vistk::process_registry::type_t const& type, types)
  {
    try
    {
      test_process(type);
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when testing "
                << "type \'" << type << "\': " << e.what() << std::endl;
    }
  }

  return 0;
}

static void test_process_configuration(vistk::process_t const process);
static void test_process_input_ports(vistk::process_t const process);
static void test_process_output_ports(vistk::process_t const process);

static void test_process_invalid_configuration(vistk::process_t const process);
static void test_process_invalid_input_port(vistk::process_t const process);
static void test_process_invalid_output_port(vistk::process_t const process);

void
test_process(vistk::process_registry::type_t const& type)
{
  static vistk::process::name_t const expected_name = vistk::process::name_t("expected_name");

  vistk::config_t config = vistk::config::empty_config();
  config->set_value(vistk::process::config_name, expected_name);

  vistk::process_registry_t const reg = vistk::process_registry::self();

  if (reg->description(type).empty())
  {
    std::cerr << "Error: The description is empty" << std::endl;
  }

  vistk::process_t const process = reg->create_process(type, config);

  if (!process)
  {
    std::cerr << "Error: Received NULL process (" << type << ")" << std::endl;

    return;
  }

  if (process->name() != expected_name)
  {
    std::cerr << "Error: Name (" << process->name() << ") "
              << "does not match expected name: " << expected_name << std::endl;
  }

  if (process->type() != type)
  {
    std::cerr << "Error: The type does not match the registry type" << std::endl;
  }

  test_process_configuration(process);
  test_process_input_ports(process);
  test_process_output_ports(process);

  test_process_invalid_configuration(process);
  test_process_invalid_input_port(process);
  test_process_invalid_output_port(process);
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
      std::cerr << "Error: Failed to get a default for "
                << process->type() << vistk::config::block_sep << key
                << ": " << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when querying for default "
                << "(" << process->type() << vistk::config::block_sep
                << key << "): " << e.what() << std::endl;
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
      std::cerr << "Error: Failed to get a info for input port "
                << process->type() << "." << port << ": " << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when querying for input port info "
                << "(" << process->type() << "." << port << "): " << e.what() << std::endl;
    }

    vistk::process::port_flags_t const& flags = info->flags;

    bool const is_const = (flags.find(vistk::process::flag_output_const) != flags.end());

    if (is_const)
    {
      std::cerr << "Error: Const flag on input port "
                << "(" << process->type() << "." << port << ")" << std::endl;
    }

    vistk::process::port_description_t const& description = info->description;

    if (description.empty())
    {
      std::cerr << "Error: Description empty on input port "
                << "(" << process->type() << "." << port << ")" << std::endl;
    }

    vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

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
      std::cerr << "Error: Failed to get a info for output port "
                << process->type() << "." << port << ": " << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when querying for output port info "
                << "(" << process->type() << "." << port << "): " << e.what() << std::endl;
    }

    vistk::process::port_flags_t const& flags = info->flags;

    bool const is_mutable = (flags.find(vistk::process::flag_input_mutable) != flags.end());

    if (is_mutable)
    {
      std::cerr << "Error: Mutable flag on output port "
                << "(" << process->type() << "." << port << ")" << std::endl;
    }

    vistk::process::port_description_t const& description = info->description;

    if (description.empty())
    {
      std::cerr << "Error: Description empty on output port "
                << "(" << process->type() << "." << port << ")" << std::endl;
    }

    vistk::edge_t edge1 = vistk::edge_t(new vistk::edge(config));
    vistk::edge_t edge2 = vistk::edge_t(new vistk::edge(config));

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

  vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

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

  vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->connect_output_port(non_existent_port, edge),
                   "requesting a connection to a non-existent output port");
}
