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

char const*
null_process_config_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
no_such_port_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
null_edge_port_connection_exception
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
          "on process \'" << m_process << "\' "
          "has already been connected to.";

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

char const*
missing_connection_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
unknown_configuration_value_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
invalid_configuration_value_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
invalid_configuration_exception
::what() const throw()
{
  return m_what.c_str();
}

}
