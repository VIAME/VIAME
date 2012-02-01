/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_exception.h"

#include <sstream>

/**
 * \file process_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::process processes\endlink.
 */

namespace vistk
{

null_process_config_exception
::null_process_config_exception() throw()
  : process_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a process.";

  m_what = sstr.str();
}

null_process_config_exception
::~null_process_config_exception() throw()
{
}

reinitialization_exception
::reinitialization_exception(process::name_t const& process) throw()
  : process_exception()
  , m_process(process)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_process << "\' "
          "was initialized a second time.";

  m_what = sstr.str();
}

reinitialization_exception
::~reinitialization_exception() throw()
{
}

null_conf_info_exception
::null_conf_info_exception(process::name_t const& process, config::key_t const& key) throw()
  : process_exception()
  , m_process(process)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_process << "\' "
          "gave NULL for the information about the "
          "configuration \'" << m_key << "\'.";

  m_what = sstr.str();
}

null_conf_info_exception
::~null_conf_info_exception() throw()
{
}

null_port_info_exception
::null_port_info_exception(process::name_t const& process, process::port_t const& port, std::string const& type) throw()
  : process_exception()
  , m_process(process)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_process << "\' "
          "gave NULL for the information about "
          "the " << type << " port "
          "\'" << m_port << "\'.";

  m_what = sstr.str();
}

null_port_info_exception
::~null_port_info_exception() throw()
{
}

null_input_port_info_exception
::null_input_port_info_exception(process::name_t const& process, process::port_t const& port) throw()
  : null_port_info_exception(process, port, "input")
{
}

null_input_port_info_exception
::~null_input_port_info_exception() throw()
{
}

null_output_port_info_exception
::null_output_port_info_exception(process::name_t const& process, process::port_t const& port) throw()
  : null_port_info_exception(process, port, "output")
{
}

null_output_port_info_exception
::~null_output_port_info_exception() throw()
{
}

uninitialized_exception
::uninitialized_exception(process::name_t const& process) throw()
  : process_exception()
  , m_process(process)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_process << "\' "
          "was stepped before initialization.";

  m_what = sstr.str();
}

uninitialized_exception
::~uninitialized_exception() throw()
{
}

port_connection_exception
::port_connection_exception(process::name_t const& process, process::port_t const& port) throw()
  : process_exception()
  , m_process(process)
  , m_port(port)
{
}

port_connection_exception
::~port_connection_exception() throw()
{
}

connect_to_initialized_process_exception
::connect_to_initialized_process_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "was requested for a connection after "
          "initialization.";

  m_what = sstr.str();
}

connect_to_initialized_process_exception
::~connect_to_initialized_process_exception() throw()
{
}

no_such_port_exception
::no_such_port_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "does not exist.";

  m_what = sstr.str();
}

no_such_port_exception
::~no_such_port_exception() throw()
{
}

null_edge_port_connection_exception
::null_edge_port_connection_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The connection to \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "was given a NULL edge.";

  m_what = sstr.str();
}

null_edge_port_connection_exception
::~null_edge_port_connection_exception() throw()
{
}

static_type_reset_exception
::static_type_reset_exception(process::name_t const& process, process::port_t const& port, process::port_type_t const& orig_type, process::port_type_t const& new_type) throw()
  : port_connection_exception(process, port)
  , m_orig_type(orig_type)
  , m_new_type(new_type)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "has the type \'" << m_orig_type << "\' "
          "and has was attempted to be set to have "
          "a type of \'" << m_new_type << "\'.";

  m_what = sstr.str();
}

static_type_reset_exception
::~static_type_reset_exception() throw()
{
}

port_reconnect_exception
::port_reconnect_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "has already been connected to.";

  m_what = sstr.str();
}

port_reconnect_exception
::~port_reconnect_exception() throw()
{
}

missing_connection_exception
::missing_connection_exception(process::name_t const& process, process::port_t const& port, std::string const& reason) throw()
  : port_connection_exception(process, port)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_process << "\' "
          "is not connected: " << m_reason << ".";

  m_what = sstr.str();
}

missing_connection_exception
::~missing_connection_exception() throw()
{
}

unknown_configuration_value_exception
::unknown_configuration_value_exception(process::name_t const& process, config::key_t const& key) throw()
  : process_configuration_exception()
  , m_process(process)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
          "on process \'" << m_process << "\' "
          "does not exist.";

  m_what = sstr.str();
}

unknown_configuration_value_exception
::~unknown_configuration_value_exception() throw()
{
}

invalid_configuration_value_exception
::invalid_configuration_value_exception(process::name_t const& process, config::key_t const& key, config::value_t const& value, config::description_t const& desc) throw()
  : process_configuration_exception()
  , m_process(process)
  , m_key(key)
  , m_value(value)
  , m_desc(desc)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
          "on process \'" << m_process << "\' "
          "was set to an invalid value \'" << m_value << "\'. "
          "A description of the value is: " << m_desc << ".";

  m_what = sstr.str();
}

invalid_configuration_value_exception
::~invalid_configuration_value_exception() throw()
{
}

invalid_configuration_exception
::invalid_configuration_exception(process::name_t const& process, std::string const& reason) throw()
  : process_configuration_exception()
  , m_process(process)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_process << "\' "
          "has a configuration issue: " << m_reason << ".";

  m_what = sstr.str();
}

invalid_configuration_exception
::~invalid_configuration_exception() throw()
{
}

}
