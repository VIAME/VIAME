// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "process_exception.h"

#include <vital/util/string.h>

#include <sstream>

/**
 * \file process_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::process processes\endlink.
 */

namespace sprokit {

// ----------------------------------------------------------------------------
process_exception
::process_exception() noexcept
  : pipeline_exception()
{
}

process_exception
::~process_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_process_config_exception
::null_process_config_exception() noexcept
  : process_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a process";

  m_what = sstr.str();
}

null_process_config_exception
::~null_process_config_exception() noexcept
{
}

// ----------------------------------------------------------------------------
already_initialized_exception
::already_initialized_exception(process::name_t const& name) noexcept
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' has already been initialized";

  m_what = sstr.str();
}

already_initialized_exception
::~already_initialized_exception() noexcept
{
}

// ----------------------------------------------------------------------------
unconfigured_exception
::unconfigured_exception(process::name_t const& name) noexcept
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' hasn\'t been configured yet";

  m_what = sstr.str();
}

unconfigured_exception
::~unconfigured_exception() noexcept
{
}

// ----------------------------------------------------------------------------
reconfigured_exception
::reconfigured_exception(process::name_t const& name) noexcept
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' was configured a second time";

  m_what = sstr.str();
}

reconfigured_exception
::~reconfigured_exception() noexcept
{
}

// ----------------------------------------------------------------------------
reinitialization_exception
::reinitialization_exception(process::name_t const& name) noexcept
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' was initialized a second time";

  m_what = sstr.str();
}

reinitialization_exception
::~reinitialization_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_conf_info_exception
::null_conf_info_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key) noexcept
  : process_exception()
  , m_name(name)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The process \'"
       << m_name
       << "\' gave NULL for the information about the configuration \'"
       << m_key << "\'";

  m_what = sstr.str();
}

null_conf_info_exception
::~null_conf_info_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_port_info_exception
::null_port_info_exception(process::name_t const& name, process::port_t const& port, std::string const& type) noexcept
  : process_exception()
  , m_name(name)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' gave NULL for the information about the "
       << type << " port \'" << m_port << "\'";

  m_what = sstr.str();
}

null_port_info_exception
::~null_port_info_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_input_port_info_exception
::null_input_port_info_exception(process::name_t const& name, process::port_t const& port) noexcept
  : null_port_info_exception(name, port, "input")
{
}

null_input_port_info_exception
::~null_input_port_info_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_output_port_info_exception
::null_output_port_info_exception(process::name_t const& name, process::port_t const& port) noexcept
  : null_port_info_exception(name, port, "output")
{
}

null_output_port_info_exception
::~null_output_port_info_exception() noexcept
{
}

// ----------------------------------------------------------------------------
flag_mismatch_exception
::flag_mismatch_exception(process::name_t const& name, process::port_t const& port, std::string const& reason) noexcept
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' gave invalid flags for the \'"
       << m_port << "\' port: "
       << m_reason;

  m_what = sstr.str();
}

flag_mismatch_exception
::~flag_mismatch_exception() noexcept
{
}

// ----------------------------------------------------------------------------
set_type_on_initialized_process_exception
::set_type_on_initialized_process_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& type) noexcept
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "The type of the port \'" << m_port << "\' on the process \'"
       << m_name << "\' was attempted to be set to \'" << m_type << "\'";

  m_what = sstr.str();
}

set_type_on_initialized_process_exception
::~set_type_on_initialized_process_exception() noexcept
{
}

// ----------------------------------------------------------------------------
set_frequency_on_initialized_process_exception
::set_frequency_on_initialized_process_exception(process::name_t const& name, process::port_t const& port, process::port_frequency_t const& frequency) noexcept
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_frequency(frequency)
{
  std::ostringstream sstr;

  sstr << "The frequency of the port \'" << m_port << "\' on the process \'"
       << m_name << "\' was attempted to be set to \'" << m_frequency << "\'";

  m_what = sstr.str();
}

set_frequency_on_initialized_process_exception
::~set_frequency_on_initialized_process_exception() noexcept
{
}

// ----------------------------------------------------------------------------
uninitialized_exception
::uninitialized_exception(process::name_t const& name) noexcept
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' was stepped before initialization";

  m_what = sstr.str();
}

uninitialized_exception
::~uninitialized_exception() noexcept
{
}

// ----------------------------------------------------------------------------
port_connection_exception
::port_connection_exception(process::name_t const& name, process::port_t const& port) noexcept
  : process_exception()
  , m_name(name)
  , m_port(port)
{
}

port_connection_exception
::~port_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
connect_to_initialized_process_exception
::connect_to_initialized_process_exception(process::name_t const& name, process::port_t const& port) noexcept
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' on process \'"
       << m_name << "\' was requested for a connection after initialization";

  m_what = sstr.str();
}

connect_to_initialized_process_exception
::~connect_to_initialized_process_exception() noexcept
{
}

// ----------------------------------------------------------------------------
no_such_port_exception
::no_such_port_exception(process::name_t const& name, process::port_t const& port) noexcept
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'"
       << m_port << "\' on process \'"
       << m_name << "\' does not exist";

  m_what = sstr.str();
}

no_such_port_exception
::no_such_port_exception(process::name_t const& name, process::port_t const& port,
                                        process::ports_t const& all_ports) noexcept
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'"
       << m_port << "\' on process \'"
       << m_name << "\' does not exist.  Available ports: "
       << kwiver::vital::join( all_ports, ", ");

  m_what = sstr.str();
}

no_such_port_exception
::~no_such_port_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_edge_port_connection_exception
::null_edge_port_connection_exception(process::name_t const& name,
                                      process::port_t const& port) noexcept
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The connection to \'" << m_port << "\' on process \'"
       << m_name << "\' was given a NULL edge";

  m_what = sstr.str();
}

null_edge_port_connection_exception
::~null_edge_port_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
static_type_reset_exception
::static_type_reset_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& orig_type, process::port_type_t const& new_type) noexcept
  : port_connection_exception(name, port)
  , m_orig_type(orig_type)
  , m_new_type(new_type)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' on process \'" << m_name
       << "\' has the type \'" << m_orig_type
       << "\' and has was attempted to be set to have a type of \'"
       << m_new_type << "\'";

  m_what = sstr.str();
}

static_type_reset_exception
::~static_type_reset_exception() noexcept
{
}

// ----------------------------------------------------------------------------
port_reconnect_exception
::port_reconnect_exception(process::name_t const& name, process::port_t const& port) noexcept
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' on process \'"
       << m_name << "\' has already been connected to";

  m_what = sstr.str();
}

port_reconnect_exception
::~port_reconnect_exception() noexcept
{
}

// ----------------------------------------------------------------------------
missing_connection_exception
::missing_connection_exception(process::name_t const& name,
                               process::port_t const& port,
                               std::string const& reason) noexcept
  : port_connection_exception(name, port)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' on process \'"
       << m_name << "\' is not connected: " << m_reason << "";

  m_what = sstr.str();
}

missing_connection_exception
::~missing_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
process_configuration_exception
::process_configuration_exception() noexcept
  : process_exception()
{
}

process_configuration_exception
::~process_configuration_exception() noexcept
{
}

// ----------------------------------------------------------------------------
unknown_configuration_value_exception
::unknown_configuration_value_exception(process::name_t const& name,
                                        kwiver::vital::config_block_key_t const& key) noexcept
  : process_configuration_exception()
  , m_name(name)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' on process \'"
       << m_name << "\' does not exist";

  m_what = sstr.str();
}

unknown_configuration_value_exception
::~unknown_configuration_value_exception() noexcept
{
}

// ----------------------------------------------------------------------------
invalid_configuration_value_exception
::invalid_configuration_value_exception(process::name_t const& name,
                                        kwiver::vital::config_block_key_t const& key,
                                        kwiver::vital::config_block_value_t const& value,
                                        kwiver::vital::config_block_description_t const& desc) noexcept
  : process_configuration_exception()
  , m_name(name)
  , m_key(key)
  , m_value(value)
  , m_desc(desc)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' on process \'" << m_name
       << "\' was set to an invalid value \'" << m_value << "\'. A description of the value is: " << m_desc;

  m_what = sstr.str();
}

invalid_configuration_value_exception
::~invalid_configuration_value_exception() noexcept
{
}

// ----------------------------------------------------------------------------
invalid_configuration_exception
::invalid_configuration_exception(process::name_t const& name, std::string const& reason) noexcept
  : process_configuration_exception()
  , m_name(name)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' has a configuration issue: " << m_reason;

  m_what = sstr.str();
}

invalid_configuration_exception
::~invalid_configuration_exception() noexcept
{
}

}
