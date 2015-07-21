/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "process_exception.h"

#include <sstream>

/**
 * \file process_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::process processes\endlink.
 */

namespace sprokit
{

process_exception
::process_exception() SPROKIT_NOTHROW
  : pipeline_exception()
{
}

process_exception
::~process_exception() SPROKIT_NOTHROW
{
}

null_process_config_exception
::null_process_config_exception() SPROKIT_NOTHROW
  : process_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a process";

  m_what = sstr.str();
}

null_process_config_exception
::~null_process_config_exception() SPROKIT_NOTHROW
{
}

already_initialized_exception
::already_initialized_exception(process::name_t const& name) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "has already been initialized";

  m_what = sstr.str();
}

already_initialized_exception
::~already_initialized_exception() SPROKIT_NOTHROW
{
}

unconfigured_exception
::unconfigured_exception(process::name_t const& name) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "hasn\'t been configured yet";

  m_what = sstr.str();
}

unconfigured_exception
::~unconfigured_exception() SPROKIT_NOTHROW
{
}

reconfigured_exception
::reconfigured_exception(process::name_t const& name) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "was configured a second time";

  m_what = sstr.str();
}

reconfigured_exception
::~reconfigured_exception() SPROKIT_NOTHROW
{
}

reinitialization_exception
::reinitialization_exception(process::name_t const& name) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "was initialized a second time";

  m_what = sstr.str();
}

reinitialization_exception
::~reinitialization_exception() SPROKIT_NOTHROW
{
}

null_conf_info_exception
::null_conf_info_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "gave NULL for the information about the "
          "configuration \'" << m_key << "\'";

  m_what = sstr.str();
}

null_conf_info_exception
::~null_conf_info_exception() SPROKIT_NOTHROW
{
}

null_port_info_exception
::null_port_info_exception(process::name_t const& name, process::port_t const& port, std::string const& type) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "gave NULL for the information about "
          "the " << type << " port "
          "\'" << m_port << "\'";

  m_what = sstr.str();
}

null_port_info_exception
::~null_port_info_exception() SPROKIT_NOTHROW
{
}

null_input_port_info_exception
::null_input_port_info_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : null_port_info_exception(name, port, "input")
{
}

null_input_port_info_exception
::~null_input_port_info_exception() SPROKIT_NOTHROW
{
}

null_output_port_info_exception
::null_output_port_info_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : null_port_info_exception(name, port, "output")
{
}

null_output_port_info_exception
::~null_output_port_info_exception() SPROKIT_NOTHROW
{
}

flag_mismatch_exception
::flag_mismatch_exception(process::name_t const& name, process::port_t const& port, std::string const& reason) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "gave invalid flags for the "
          "\'" << m_port << "\' port: "
       << m_reason;

  m_what = sstr.str();
}

flag_mismatch_exception
::~flag_mismatch_exception() SPROKIT_NOTHROW
{
}

set_type_on_initialized_process_exception
::set_type_on_initialized_process_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& type) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "The type of the port \'" << m_port << "\' "
          "on the process \'" << m_name << "\' was "
          "attempted to be set to \'" << m_type << "\'";

  m_what = sstr.str();
}

set_type_on_initialized_process_exception
::~set_type_on_initialized_process_exception() SPROKIT_NOTHROW
{
}

set_frequency_on_initialized_process_exception
::set_frequency_on_initialized_process_exception(process::name_t const& name, process::port_t const& port, process::port_frequency_t const& frequency) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_port(port)
  , m_frequency(frequency)
{
  std::ostringstream sstr;

  sstr << "The frequency of the port \'" << m_port << "\' "
          "on the process \'" << m_name << "\' was "
          "attempted to be set to \'" << m_frequency << "\'";

  m_what = sstr.str();
}

set_frequency_on_initialized_process_exception
::~set_frequency_on_initialized_process_exception() SPROKIT_NOTHROW
{
}

uninitialized_exception
::uninitialized_exception(process::name_t const& name) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "was stepped before initialization";

  m_what = sstr.str();
}

uninitialized_exception
::~uninitialized_exception() SPROKIT_NOTHROW
{
}

port_connection_exception
::port_connection_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : process_exception()
  , m_name(name)
  , m_port(port)
{
}

port_connection_exception
::~port_connection_exception() SPROKIT_NOTHROW
{
}

connect_to_initialized_process_exception
::connect_to_initialized_process_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "was requested for a connection after "
          "initialization";

  m_what = sstr.str();
}

connect_to_initialized_process_exception
::~connect_to_initialized_process_exception() SPROKIT_NOTHROW
{
}

no_such_port_exception
::no_such_port_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "does not exist";

  m_what = sstr.str();
}

no_such_port_exception
::~no_such_port_exception() SPROKIT_NOTHROW
{
}

null_edge_port_connection_exception
::null_edge_port_connection_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The connection to \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "was given a NULL edge";

  m_what = sstr.str();
}

null_edge_port_connection_exception
::~null_edge_port_connection_exception() SPROKIT_NOTHROW
{
}

static_type_reset_exception
::static_type_reset_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& orig_type, process::port_type_t const& new_type) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
  , m_orig_type(orig_type)
  , m_new_type(new_type)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "has the type \'" << m_orig_type << "\' "
          "and has was attempted to be set to have "
          "a type of \'" << m_new_type << "\'";

  m_what = sstr.str();
}

static_type_reset_exception
::~static_type_reset_exception() SPROKIT_NOTHROW
{
}

port_reconnect_exception
::port_reconnect_exception(process::name_t const& name, process::port_t const& port) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "has already been connected to";

  m_what = sstr.str();
}

port_reconnect_exception
::~port_reconnect_exception() SPROKIT_NOTHROW
{
}

missing_connection_exception
::missing_connection_exception(process::name_t const& name, process::port_t const& port, std::string const& reason) SPROKIT_NOTHROW
  : port_connection_exception(name, port)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The port \'" << m_port << "\' "
          "on process \'" << m_name << "\' "
          "is not connected: " << m_reason << "";

  m_what = sstr.str();
}

missing_connection_exception
::~missing_connection_exception() SPROKIT_NOTHROW
{
}

process_configuration_exception
::process_configuration_exception() SPROKIT_NOTHROW
  : process_exception()
{
}

process_configuration_exception
::~process_configuration_exception() SPROKIT_NOTHROW
{
}

unknown_configuration_value_exception
::unknown_configuration_value_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key) SPROKIT_NOTHROW
  : process_configuration_exception()
  , m_name(name)
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
          "on process \'" << m_name << "\' "
          "does not exist";

  m_what = sstr.str();
}

unknown_configuration_value_exception
::~unknown_configuration_value_exception() SPROKIT_NOTHROW
{
}

invalid_configuration_value_exception
::invalid_configuration_value_exception(process::name_t const& name,
                                        kwiver::vital::config_block_key_t const& key,
                                        kwiver::vital::config_block_value_t const& value,
                                        kwiver::vital::config_block_description_t const& desc) SPROKIT_NOTHROW
  : process_configuration_exception()
  , m_name(name)
  , m_key(key)
  , m_value(value)
  , m_desc(desc)
{
  std::ostringstream sstr;

  sstr << "The configuration value \'" << m_key << "\' "
          "on process \'" << m_name << "\' "
          "was set to an invalid value \'" << m_value << "\'. "
          "A description of the value is: " << m_desc << "";

  m_what = sstr.str();
}

invalid_configuration_value_exception
::~invalid_configuration_value_exception() SPROKIT_NOTHROW
{
}

invalid_configuration_exception
::invalid_configuration_exception(process::name_t const& name, std::string const& reason) SPROKIT_NOTHROW
  : process_configuration_exception()
  , m_name(name)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The process \'" << m_name << "\' "
          "has a configuration issue: " << m_reason;

  m_what = sstr.str();
}

invalid_configuration_exception
::~invalid_configuration_exception() SPROKIT_NOTHROW
{
}

}
