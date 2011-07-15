/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_exception.h"

#include <sstream>

namespace vistk
{

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

no_such_port_exception
::no_such_port_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
       << "on process \'" << m_process << "\' "
       << "does not exist.";

  m_what = sstr.str();
}

no_such_port_exception
::~no_such_port_exception() throw()
{
}

char const*
no_such_port_exception
::what() const throw()
{
  return m_what.c_str();
}

null_edge_port_connection
::null_edge_port_connection(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The connection to \'" << m_port << "\' "
       << "on process \'" << m_process << "\' "
       << "was given a NULL edge.";

  m_what = sstr.str();
}

null_edge_port_connection
::~null_edge_port_connection() throw()
{
}

char const*
null_edge_port_connection
::what() const throw()
{
  return m_what.c_str();
}

port_reconnect_exception
::port_reconnect_exception(process::name_t const& process, process::port_t const& port) throw()
  : port_connection_exception(process, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
       << "on process \'" << m_process << "\' "
       << "has already been connected to.";

  m_what = sstr.str();
}

port_reconnect_exception
::~port_reconnect_exception() throw()
{
}

char const*
port_reconnect_exception
::what() const throw()
{
  return m_what.c_str();
}

unknown_configuration_value
::unknown_configuration_value(process::name_t const& process, config::key_t const& key) throw()
  : process_exception()
  , m_process(process)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
       << "on process \'" << m_process << "\' "
       << "does not exist.";

  m_what = sstr.str();
}

unknown_configuration_value
::~unknown_configuration_value() throw()
{
}

char const*
unknown_configuration_value
::what() const throw()
{
  return m_what.c_str();
}

invalid_configuration_value
::invalid_configuration_value(process::name_t const& process, config::key_t const& key, config::value_t const& value, config::description_t const& desc) throw()
  : process_exception()
  , m_process(process)
  , m_key(key)
  , m_value(value)
  , m_desc(desc)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
       << "on process \'" << m_process << "\' "
       << "was set to an invalid value \'" << m_value << "\'. "
       << "A description of the value is: " << m_desc << ".";

  m_what = sstr.str();
}

invalid_configuration_value
::~invalid_configuration_value() throw()
{
}

char const*
invalid_configuration_value
::what() const throw()
{
  return m_what.c_str();
}

broken_pass_through_exception
::broken_pass_through_exception(process::name_t const& process, process::port_t const& port, std::string const& type) throw()
  : process_exception()
  , m_process(process)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The pass-through port \'" << m_port << "\' "
       << "on process \'" << m_process << "\' "
       << "is missing an " << type << " connection.";

  m_what = sstr.str();
}

broken_pass_through_exception
::~broken_pass_through_exception() throw()
{
}

char const*
broken_pass_through_exception
::what() const throw()
{
  return m_what.c_str();
}

missing_input_pass_through
::missing_input_pass_through(process::name_t const& process, process::name_t const& new_process) throw()
  : broken_pass_through_exception(process, new_process, "input")
{
}

missing_input_pass_through
::~missing_input_pass_through() throw()
{
}

missing_output_pass_through
::missing_output_pass_through(process::name_t const& process, process::name_t const& new_process) throw()
  : broken_pass_through_exception(process, new_process, "output")
{
}

missing_output_pass_through
::~missing_output_pass_through() throw()
{
}

} // end namespace vistk
