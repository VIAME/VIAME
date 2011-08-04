/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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

int
main()
{
  vistk::process_registry_t reg = vistk::process_registry::self();

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

  vistk::process::name_t const expected_name = vistk::process::name_t("expected_name");

  vistk::process_registry::types_t const types = reg->types();

  vistk::config_t config = vistk::config::empty_config();
  config->set_value(vistk::process::config_name, expected_name);

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
    }

    if (!process)
    {
      std::cerr << "Error: Received NULL process (" << type << ")" << std::endl;

      continue;
    }

    if (reg->description(type).empty())
    {
      std::cerr << "Error: The description is empty" << std::endl;
    }

    if (process->name() != expected_name)
    {
      std::cerr << "Error: Name (" << process->name() << ") "
                << "does not match expected name: " << expected_name << std::endl;
    }

    if (process->type().empty())
    {
      std::cerr << "Error: The type is empty" << std::endl;
    }

    vistk::config::keys_t const keys = process->available_config();

    BOOST_FOREACH (vistk::config::key_t const& key, keys)
    {
      try
      {
        process->config_default(key);
      }
      catch (vistk::unknown_configuration_value_exception& e)
      {
        std::cerr << "Error: Failed to get a default for "
                  << type << vistk::config::block_sep << key
                  << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for default "
                  << "(" << type << vistk::config::block_sep
                  << key << "): " << e.what() << std::endl;
      }

      try
      {
        process->config_description(key);
      }
      catch (vistk::unknown_configuration_value_exception& e)
      {
        std::cerr << "Error: Failed to get a description for "
                  << type << vistk::config::block_sep << key
                  << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for description "
                  << "(" << type << vistk::config::block_sep
                  << key << "): " << e.what() << std::endl;
      }
    }

    vistk::process::ports_t const iports = process->input_ports();

    BOOST_FOREACH (vistk::process::port_t const& port, iports)
    {
      try
      {
        process->input_port_type(port);
      }
      catch (vistk::no_such_port_exception& e)
      {
        std::cerr << "Error: Failed to get a type for input port "
                  << type << "." << port << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for input port type "
                  << "(" << type << "." << port << "): " << e.what() << std::endl;
      }

      try
      {
        process->input_port_description(port);
      }
      catch (vistk::no_such_port_exception& e)
      {
        std::cerr << "Error: Failed to get a description for input port "
                  << type << "." << port << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for input port description "
                  << "(" << type << "." << port << "): " << e.what() << std::endl;
      }
    }

    vistk::process::ports_t const oports = process->output_ports();

    BOOST_FOREACH (vistk::process::port_t const& port, oports)
    {
      try
      {
        process->output_port_type(port);
      }
      catch (vistk::no_such_port_exception& e)
      {
        std::cerr << "Error: Failed to get a type for output port "
                  << type << "." << port << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for output port type "
                  << "(" << type << "." << port << "): " << e.what() << std::endl;
      }

      try
      {
        process->output_port_description(port);
      }
      catch (vistk::no_such_port_exception& e)
      {
        std::cerr << "Error: Failed to get a description for output port "
                  << type << "." << port << ": " << e.what() << std::endl;
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when querying for output port description "
                  << "(" << type << "." << port << "): " << e.what() << std::endl;
      }
    }

    // Test for proper exceptions on invalid config requests.
    {
      vistk::config::key_t const non_existent_config = vistk::config::key_t("does_not_exist");

      try
      {
        process->config_default(non_existent_config);
      }
      catch (vistk::unknown_configuration_value_exception& e)
      {
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when "
                  << "requesting default for non-existent config: "
                  << e.what() << std::endl;
      }

      try
      {
        process->config_description(non_existent_config);
      }
      catch (vistk::unknown_configuration_value_exception& e)
      {
      }
      catch (std::exception& e)
      {
        std::cerr << "Error: Unexpected exception when "
                  << "requesting description for non-existent config: "
                  << e.what() << std::endl;
      }
    }

    // Test for proper exceptions on invalid port requests.
    {
      vistk::process::port_t const non_existent_port = vistk::process::port_t("does_not_exist");

      // Input ports.
      {
        try
        {
          process->input_port_type(non_existent_port);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting type on non-existent input port: "
                    << e.what() << std::endl;
        }

        try
        {
          process->input_port_description(non_existent_port);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting description for non-existent input port: "
                    << e.what() << std::endl;
        }

        vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

        try
        {
          process->connect_input_port(non_existent_port, edge);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting connection to non-existent input port: "
                    << e.what() << std::endl;
        }
      }

      // Output ports.
      {
        try
        {
          process->output_port_type(non_existent_port);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting type on non-existent output port: "
                    << e.what() << std::endl;
        }

        try
        {
          process->output_port_description(non_existent_port);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting description for non-existent output port: "
                    << e.what() << std::endl;
        }

        vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

        try
        {
          process->connect_output_port(non_existent_port, edge);
        }
        catch (vistk::no_such_port_exception& e)
        {
        }
        catch (std::exception& e)
        {
          std::cerr << "Error: Unexpected exception when "
                    << "requesting connection to non-existent output port: "
                    << e.what() << std::endl;
        }
      }
    }
  }

  // Check exceptions for unknown types.
  {
    vistk::process_registry::type_t const non_existent_process = vistk::process_registry::type_t("no_such_process");

    try
    {
      reg->create_process(non_existent_process, config);
    }
    catch (vistk::no_such_process_type_exception& e)
    {
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when "
                << "requesting a non-existent process type: "
                << e.what() << std::endl;
    }

    try
    {
      reg->description(non_existent_process);
    }
    catch (vistk::no_such_process_type_exception& e)
    {
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when "
                << "requesting a description of a non-existent process type: "
                << e.what() << std::endl;
    }
  }

  return 0;
}
