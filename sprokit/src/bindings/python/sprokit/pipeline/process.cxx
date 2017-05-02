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

#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/stamp.h>

#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>
#include <sprokit/python/util/python_wrap_const_shared_ptr.h>
#include <sprokit/python/util/set_indexing_suite.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/operators.hpp>

/**
 * \file process.cxx
 *
 * \brief Python bindings for \link sprokit::process\endlink.
 */

using namespace boost::python;

/// \todo How to do grab_input_as<>?

class wrap_process
  : public sprokit::process
  , public wrapper<sprokit::process>
{
  public:
    wrap_process(kwiver::vital::config_block_sptr const& config);
    ~wrap_process();

    void _base_configure();

    void _base_init();

    void _base_reset();

    void _base_flush();

    void _base_step();

    void _base_reconfigure(kwiver::vital::config_block_sptr const& conf);

    properties_t _base_properties() const;

    ports_t _base_input_ports() const;
    ports_t _base_output_ports() const;

    port_info_t _base_input_port_info(port_t const& port);
    port_info_t _base_output_port_info(port_t const& port);

    bool _base_set_input_port_type(port_t const& port, port_type_t const& new_type);
    bool _base_set_output_port_type(port_t const& port, port_type_t const& new_type);

    kwiver::vital::config_block_keys_t _base_available_config() const;

    conf_info_t _base_config_info(kwiver::vital::config_block_key_t const& key);

    void _configure();

    void _init();

    void _reset();

    void _flush();

    void _step();

    void _reconfigure(kwiver::vital::config_block_sptr const& conf);

    properties_t _properties() const;

    ports_t _input_ports() const;
    ports_t _output_ports() const;

    port_info_t _input_port_info(port_t const& port);
    port_info_t _output_port_info(port_t const& port);

    bool _set_input_port_type(port_t const& port, port_type_t const& new_type);
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type);

    kwiver::vital::config_block_keys_t _available_config() const;

    conf_info_t _config_info(kwiver::vital::config_block_key_t const& key);

    void _declare_input_port(port_t const& port, port_info_t const& info);
    void _declare_input_port_1(port_t const& port,
                               port_type_t const& type_,
                               port_flags_t const& flags_,
                               port_description_t const& description_,
                               port_frequency_t const& frequency_);
    void _declare_output_port(port_t const& port, port_info_t const& info);
    void _declare_output_port_1(port_t const& port,
                                port_type_t const& type_,
                                port_flags_t const& flags_,
                                port_description_t const& description_,
                                port_frequency_t const& frequency_);

    void _set_input_port_frequency(port_t const& port, port_frequency_t const& new_frequency);
    void _set_output_port_frequency(port_t const& port, port_frequency_t const& new_frequency);

    void _remove_input_port(port_t const& port);
    void _remove_output_port(port_t const& port);

    void _declare_configuration_key(kwiver::vital::config_block_key_t const& key, conf_info_t const& info);
    void _declare_configuration_key_1(kwiver::vital::config_block_key_t const& key,
                                      kwiver::vital::config_block_value_t const& def_,
                                      kwiver::vital::config_block_description_t const& description_);
    void _declare_configuration_key_2(kwiver::vital::config_block_key_t const& key,
                                      kwiver::vital::config_block_value_t const& def_,
                                      kwiver::vital::config_block_description_t const& description_,
                                      bool tunable_);

    void _mark_process_as_complete();

    bool _has_input_port_edge(port_t const& port) const;
    size_t _count_output_port_edges(port_t const& port) const;

    sprokit::edge_datum_t _peek_at_port(port_t const& port, size_t idx) const;
    sprokit::datum_t _peek_at_datum_on_port(port_t const& port, size_t idx) const;
    sprokit::edge_datum_t _grab_from_port(port_t const& port) const;
    sprokit::datum_t _grab_datum_from_port(port_t const& port) const;
    object _grab_value_from_port(port_t const& port) const;
    void _push_to_port(port_t const& port, sprokit::edge_datum_t const& dat) const;
    void _push_datum_to_port(port_t const& port, sprokit::datum_t const& dat) const;
    void _push_value_to_port(port_t const& port, object const& obj) const;

    kwiver::vital::config_block_sptr _get_config() const;
    kwiver::vital::config_block_value_t _config_value(kwiver::vital::config_block_key_t const& key) const;

    void _set_data_checking_level(data_check_t check);

    data_info_t _edge_data_info(sprokit::edge_data_t const& data);
};

BOOST_PYTHON_MODULE(process)
{
  class_<sprokit::process::name_t>("ProcessName"
    , "A type for the name of a process.");
  class_<sprokit::process::names_t>("ProcessNames"
    , "A collection of process names.")
    .def(vector_indexing_suite<sprokit::process::names_t>())
  ;
  class_<sprokit::process::type_t>("ProcessType"
    , "The type for a type of process.");
  class_<sprokit::process::types_t>("ProcessTypes"
    , "A collection of process types.")
    .def(vector_indexing_suite<sprokit::process::types_t>())
  ;
  class_<sprokit::process::property_t>("ProcessProperty"
    , "A property on a process.");
  class_<sprokit::process::properties_t>("ProcessProperties"
    , "A collection of properties on a process.")
    .def(set_indexing_suite<sprokit::process::properties_t>())
  ;
  class_<sprokit::process::port_description_t>("PortDescription"
    , "A description for a port.");
  class_<sprokit::process::port_t>("Port"
    , "The name of a port.");
  class_<sprokit::process::ports_t>("Ports"
    , "A collection of ports.")
    .def(vector_indexing_suite<sprokit::process::ports_t>())
  ;
  class_<sprokit::process::port_type_t>("PortType"
    , "The type of data on a port.");
  class_<sprokit::process::port_flag_t>("PortFlag"
    , "A flag on a port.");
  class_<sprokit::process::port_flags_t>("PortFlags"
    , "A collection of port flags.")
    .def(set_indexing_suite<sprokit::process::port_flags_t>())
  ;
  class_<sprokit::process::port_frequency_t>("PortFrequency"
    , "A frequency for a port."
    , no_init)
    .def(init<sprokit::process::frequency_component_t>())
    .def(init<sprokit::process::frequency_component_t, sprokit::process::frequency_component_t>())
    .def("numerator", &sprokit::process::port_frequency_t::numerator
      , "The numerator of the frequency.")
    .def("denominator", &sprokit::process::port_frequency_t::denominator
      , "The denominator of the frequency.")
    .def(self <  self)
    .def(self <= self)
    .def(self == self)
    .def(self >= self)
    .def(self >  self)
    .def(self + self)
    .def(self - self)
    .def(self * self)
    .def(self / self)
    .def(!self)
  ;
  class_<sprokit::process::port_addr_t>("PortAddr"
    , "An address for a port within a pipeline.")
    .def_readwrite("process", &sprokit::process::port_addr_t::first)
    .def_readwrite("port", &sprokit::process::port_addr_t::second)
  ;
  class_<sprokit::process::port_addrs_t>("PortAddrs"
    , "A collection of port addresses.")
    .def(vector_indexing_suite<sprokit::process::port_addrs_t>())
  ;
  class_<sprokit::process::connection_t>("Connection"
    , "A connection between two ports.")
    .def_readwrite("upstream", &sprokit::process::connection_t::first)
    .def_readwrite("downstream", &sprokit::process::connection_t::second)
  ;
  class_<sprokit::process::connections_t>("Connections"
    , "A collection of connections.")
    .def(vector_indexing_suite<sprokit::process::connections_t>())
  ;

  class_<sprokit::process::port_info, sprokit::process::port_info_t>("PortInfo"
    , "Information about a port on a process."
    , no_init)
    .def(init<sprokit::process::port_type_t, sprokit::process::port_flags_t, sprokit::process::port_description_t, sprokit::process::port_frequency_t>())
    .def_readonly("type", &sprokit::process::port_info::type)
    .def_readonly("flags", &sprokit::process::port_info::flags)
    .def_readonly("description", &sprokit::process::port_info::description)
    .def_readonly("frequency", &sprokit::process::port_info::frequency)
  ;

  implicitly_convertible<std::shared_ptr<sprokit::process::port_info>, sprokit::process::port_info_t>();

  class_<sprokit::process::conf_info, sprokit::process::conf_info_t>("ConfInfo"
    , "Information about a configuration on a process."
    , no_init)
    .def(init<kwiver::vital::config_block_value_t, kwiver::vital::config_block_description_t, bool>())
    .def_readonly("default", &sprokit::process::conf_info::def)
    .def_readonly("description", &sprokit::process::conf_info::description)
    .def_readonly("tunable", &sprokit::process::conf_info::tunable)
  ;

  implicitly_convertible<std::shared_ptr<sprokit::process::conf_info>, sprokit::process::conf_info_t>();

  class_<sprokit::process::data_info, sprokit::process::data_info_t>("DataInfo"
    , "Information about a set of data packets from edges."
    , no_init)
    .def(init<bool, sprokit::datum::type_t>())
    .def_readonly("in_sync", &sprokit::process::data_info::in_sync)
    .def_readonly("max_status", &sprokit::process::data_info::max_status)
  ;

  enum_<sprokit::process::data_check_t>("DataCheck"
    , "Levels of input validation")
    .value("none", sprokit::process::check_none)
    .value("sync", sprokit::process::check_sync)
    .value("valid", sprokit::process::check_valid)
  ;

  implicitly_convertible<std::shared_ptr<sprokit::process::data_info>, sprokit::process::data_info_t>();

  class_<wrap_process, boost::noncopyable>("PythonProcess"
    , "The base class for Python processes."
    , no_init)
    .def(init<kwiver::vital::config_block_sptr>())
    .def("configure", &sprokit::process::configure
      , "Configure the process.")
    .def("init", &sprokit::process::init
      , "Initializes the process.")
    .def("reset", &sprokit::process::reset
      , "Resets the process.")
    .def("step", &sprokit::process::step
      , "Steps the process for one iteration.")
    .def("properties", &sprokit::process::properties
      , "Returns the properties on the process.")
    .def("connect_input_port", &sprokit::process::connect_input_port
      , (arg("port"), arg("edge"))
      , "Connects the given edge to the input port.")
    .def("connect_output_port", &sprokit::process::connect_output_port
      , (arg("port"), arg("edge"))
      , "Connects the given edge to the output port.")
    .def("input_ports", &sprokit::process::input_ports
      , "Returns a list of input ports on the process.")
    .def("output_ports", &sprokit::process::output_ports
      , "Returns a list of output ports on the process.")
    .def("input_port_info", &sprokit::process::input_port_info
      , (arg("port"))
      , "Returns information about the given input port.")
    .def("output_port_info", &sprokit::process::output_port_info
      , (arg("port"))
      , "Returns information about the given output port.")
    .def("set_input_port_type", &sprokit::process::set_input_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an input port.")
    .def("set_output_port_type", &sprokit::process::set_output_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an output port.")
    .def("available_config", &sprokit::process::available_config
      , "Returns a list of available configuration keys for the process.")
    .def("available_tunable_config", &sprokit::process::available_tunable_config
      , "Returns a list of available tunable configuration keys for the process.")
    .def("config_info", &sprokit::process::config_info
      , (arg("config"))
      , "Returns information about the given configuration key.")
    .def("name", &sprokit::process::name
      , "Returns the name of the process.")
    .def("type", &sprokit::process::type
      , "Returns the type of the process.")
    .def_readonly("property_no_threads", &sprokit::process::property_no_threads)
    .def_readonly("property_no_reentrancy", &sprokit::process::property_no_reentrancy)
    .def_readonly("property_unsync_input", &sprokit::process::property_unsync_input)
    .def_readonly("property_unsync_output", &sprokit::process::property_unsync_output)
    .def_readonly("port_heartbeat", &sprokit::process::port_heartbeat)
    .def_readonly("config_name", &sprokit::process::config_name)
    .def_readonly("config_type", &sprokit::process::config_type)
    .def_readonly("type_any", &sprokit::process::type_any)
    .def_readonly("type_none", &sprokit::process::type_none)
    .def_readonly("type_data_dependent", &sprokit::process::type_data_dependent)
    .def_readonly("type_flow_dependent", &sprokit::process::type_flow_dependent)
    .def_readonly("flag_output_const", &sprokit::process::flag_output_const)
    .def_readonly("flag_output_shared", &sprokit::process::flag_output_shared)
    .def_readonly("flag_input_static", &sprokit::process::flag_input_static)
    .def_readonly("flag_input_mutable", &sprokit::process::flag_input_mutable)
    .def_readonly("flag_input_nodep", &sprokit::process::flag_input_nodep)
    .def_readonly("flag_required", &sprokit::process::flag_required)
    .def("_base_configure", &wrap_process::_base_configure
      , "Base class configure.")
    .def("_base_init", &wrap_process::_base_init
      , "Base class initialization.")
    .def("_base_reset", &wrap_process::_base_reset
      , "Base class reset.")
    .def("_base_flush", &wrap_process::_base_flush
      , "Base class flush.")
    .def("_base_step", &wrap_process::_base_step
      , "Base class step.")
    .def("_base_reconfigure", &wrap_process::_base_reconfigure
      , (arg("config"))
      , "Base class reconfigure.")
    .def("_base_properties", &wrap_process::_base_properties
      , "Base class properties.")
    .def("_base_input_ports", &wrap_process::_base_input_ports
      , "Base class input ports.")
    .def("_base_output_ports", &wrap_process::_base_output_ports
      , "Base class output ports.")
    .def("_base_input_port_info", &wrap_process::_base_input_port_info
      , (arg("port"))
      , "Base class input port info.")
    .def("_base_output_port_info", &wrap_process::_base_output_port_info
      , (arg("port"))
      , "Base class output port info.")
    .def("_base_input_port_info", &wrap_process::_base_input_port_info
      , (arg("port"), arg("new_type"))
      , "Base class input port type setting.")
    .def("_base_output_port_info", &wrap_process::_base_output_port_info
      , (arg("port"), arg("new_type"))
      , "Base class output port type setting.")
    .def("_base_set_input_port_type", &wrap_process::_base_set_input_port_type
      , (arg("port"), arg("new_type"))
      , "Base class input port type setting.")
    .def("_base_set_output_port_type", &wrap_process::_base_set_output_port_type
      , (arg("port"), arg("new_type"))
      , "Base class output port type setting.")
    .def("_base_available_config", &wrap_process::_base_available_config
      , "Base class available configuration information.")
    .def("_base_config_info", &wrap_process::_base_config_info
      , (arg("config"))
      , "Base class configuration information.")
    .def("_configure", &wrap_process::_configure, &wrap_process::_base_configure
      , "Configures the process subclass.")
    .def("_init", &wrap_process::_init, &wrap_process::_base_init
      , "Initializes the process subclass.")
    .def("_reset", &wrap_process::_reset, &wrap_process::_base_reset
      , "Resets the process subclass.")
    .def("_flush", &wrap_process::_flush, &wrap_process::_base_flush
      , "Flushes the process subclass.")
    .def("_step", &wrap_process::_step, &wrap_process::_base_step
      , "Step the process subclass for one iteration.")
    .def("_reconfigure", &wrap_process::_reconfigure, &wrap_process::_base_reconfigure
      , (arg("config"))
      , "Runtime configuration for subclasses.")
    .def("_properties", &wrap_process::_properties, &wrap_process::_base_properties
      , "The properties on the subclass.")
    .def("_input_ports", &wrap_process::_input_ports, &wrap_process::_base_input_ports
      , "Returns a list on input ports on the subclass process.")
    .def("_output_ports", &wrap_process::_output_ports, &wrap_process::_base_output_ports
      , "Returns a list on output ports on the subclass process.")
    .def("_input_port_info", &wrap_process::_input_port_info, &wrap_process::_base_input_port_info
      , (arg("port"))
      , "Returns information about the given subclass input port.")
    .def("_output_port_info", &wrap_process::_output_port_info, &wrap_process::_base_output_port_info
      , (arg("port"))
      , "Returns information about the given subclass output port.")
    .def("_set_input_port_type", &wrap_process::_set_input_port_type, &wrap_process::_base_set_input_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an input port.")
    .def("_set_output_port_type", &wrap_process::_set_output_port_type, &wrap_process::_base_set_output_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an output port.")
    .def("_available_config", &wrap_process::_available_config, &wrap_process::_base_available_config
      , "Returns a list of available configuration keys for the subclass process.")
    .def("_config_info", &wrap_process::_config_info, &wrap_process::_base_config_info
      , (arg("key"))
      , "Returns information about the given configuration key.")
    .def("declare_input_port", &wrap_process::_declare_input_port
      , (arg("port"), arg("info"))
      , "Declare an input port on the process.")
    .def("declare_input_port", &wrap_process::_declare_input_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1))
      , "Declare an input port on the process.")
    .def("declare_output_port", &wrap_process::_declare_output_port
      , (arg("port"), arg("info"))
      , "Declare an output port on the process.")
    .def("declare_output_port", &wrap_process::_declare_output_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1))
      , "Declare an output port on the process.")
    .def("set_input_port_frequency", &wrap_process::_set_input_port_frequency
      , (arg("port"), arg("new_frequency"))
      , "Set an input port\'s frequency")
    .def("set_output_port_frequency", &wrap_process::_set_output_port_frequency
      , (arg("port"), arg("new_frequency"))
      , "Set an output port\'s frequency")
    .def("remove_input_port", &wrap_process::_remove_input_port
      , (arg("port"))
      , "Remove an input port from the process.")
    .def("remove_output_port", &wrap_process::_remove_output_port
      , (arg("port"))
      , "Remove an output port from the process.")
    .def("declare_configuration_key", &wrap_process::_declare_configuration_key
      , (arg("key"), arg("info"))
      , "Declare a configuration key for the process.")
    .def("declare_configuration_key", &wrap_process::_declare_configuration_key_1
      , (arg("key"), arg("default"), arg("description"))
      , "Declare a configuration key for the process.")
    .def("declare_configuration_key", &wrap_process::_declare_configuration_key_2
      , (arg("key"), arg("default"), arg("description"), arg("tunable"))
      , "Declare a configuration key for the process.")
    .def("mark_process_as_complete", &wrap_process::_mark_process_as_complete
      , "Tags the process as complete.")
    .def("has_input_port_edge", &wrap_process::_has_input_port_edge
      , (arg("port"))
      , "True if there is an edge that is connected to the port, False otherwise.")
    .def("count_output_port_edges", &wrap_process::_count_output_port_edges
      , (arg("port"))
      , "The number of edges that are connected to the port.")
    .def("peek_at_port", &wrap_process::_peek_at_port
      , (arg("port"), arg("idx") = 0)
      , "Peek at a port.")
    .def("peek_at_datum_on_port", &wrap_process::_peek_at_datum_on_port
      , (arg("port"), arg("idx") = 0)
      , "Peek at a datum on a port.")
    .def("grab_from_port", &wrap_process::_grab_from_port
      , (arg("port"))
      , "Grab a datum packet from a port.")
    .def("grab_value_from_port", &wrap_process::_grab_value_from_port
      , (arg("port"))
      , "Grab a value from a port.")
    .def("grab_datum_from_port", &wrap_process::_grab_datum_from_port
      , (arg("port"))
      , "Grab a datum from a port.")
    .def("push_to_port", &wrap_process::_push_to_port
      , (arg("port"), arg("datum"))
      , "Push a datum packet to a port.")
    .def("push_value_to_port", &wrap_process::_push_value_to_port
      , (arg("port"), arg("value"))
      , "Push a value to a port.")
    .def("push_datum_to_port", &wrap_process::_push_datum_to_port
      , (arg("port"), arg("datum"))
      , "Push a datum to a port.")
    .def("get_config", &wrap_process::_get_config
      , "Gets the configuration for the process.")
    .def("config_value", &wrap_process::_config_value
      , (arg("key"))
      , "Gets a value from the configuration for the process.")
    .def("set_data_checking_level", &wrap_process::_set_data_checking_level
      , (arg("check"))
      , "Set the level to which the inputs are automatically checked.")
    .def("edge_data_info", &wrap_process::_edge_data_info
      , (arg("data"))
      , "Returns information about the given data.")
  ;
}

wrap_process
::wrap_process(kwiver::vital::config_block_sptr const& config)
  : sprokit::process(config)
{
}

wrap_process
::~wrap_process()
{
}

void
wrap_process
::_base_configure()
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_configure())
}

void
wrap_process
::_base_init()
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_init())
}

void
wrap_process
::_base_reset()
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_reset())
}

void
wrap_process
::_base_flush()
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_flush())
}

void
wrap_process
::_base_step()
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_step())
}

void
wrap_process
::_base_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(process::_reconfigure(conf))
}

sprokit::process::properties_t
wrap_process
::_base_properties() const
{
  static sprokit::process::property_t const property_python = sprokit::process::property_t("_python");

  properties_t consts = process::_properties();

  consts.insert(property_python);

  return consts;
}

sprokit::process::ports_t
wrap_process
::_base_input_ports() const
{
  return process::_input_ports();
}

sprokit::process::ports_t
wrap_process
::_base_output_ports() const
{
  return process::_output_ports();
}

sprokit::process::port_info_t
wrap_process
::_base_input_port_info(port_t const& port)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_input_port_info(port))
}

sprokit::process::port_info_t
wrap_process
::_base_output_port_info(port_t const& port)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_output_port_info(port))
}

bool
wrap_process
::_base_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_set_input_port_type(port, new_type))
}

bool
wrap_process
::_base_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_set_input_port_type(port, new_type))
}

kwiver::vital::config_block_keys_t
wrap_process
::_base_available_config() const
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_available_config())
}

sprokit::process::conf_info_t
wrap_process
::_base_config_info(kwiver::vital::config_block_key_t const& key)
{
  SPROKIT_PYTHON_TRANSLATE_EXCEPTION(return process::_config_info(key))
}

void
wrap_process
::_configure()
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_configure");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f())

      return;
    }
  }

  _base_configure();
}

void
wrap_process
::_init()
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_init");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f())

      return;
    }
  }

  _base_init();
}

void
wrap_process
::_reset()
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_reset");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f())

      return;
    }
  }

  _base_reset();
}

void
wrap_process
::_flush()
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_flush");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f())

      return;
    }
  }

  _base_flush();
}

void
wrap_process
::_step()
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_step");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f())

      return;
    }
  }

  _base_step();
}

void
wrap_process
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_reconfigure");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(f(conf))

      return;
    }
  }

  _base_reconfigure(conf);
}

sprokit::process::properties_t
wrap_process
::_properties() const
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_properties");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f())
    }
  }

  return _base_properties();
}

sprokit::process::ports_t
wrap_process
::_input_ports() const
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_input_ports");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f())
    }
  }

  return _base_input_ports();
}

sprokit::process::ports_t
wrap_process
::_output_ports() const
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_output_ports");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f())
    }
  }

  return _base_output_ports();
}

sprokit::process::port_info_t
wrap_process
::_input_port_info(port_t const& port)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_input_port_info");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f(port))
    }
  }

  return _base_input_port_info(port);
}

sprokit::process::port_info_t
wrap_process
::_output_port_info(port_t const& port)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_output_port_info");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f(port))
    }
  }

  return _base_output_port_info(port);
}

bool
wrap_process
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_set_input_port_type");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f(port, new_type))
    }
  }

  return _base_set_input_port_type(port, new_type);
}

bool
wrap_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_set_output_port_type");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f(port, new_type))
    }
  }

  return _base_set_output_port_type(port, new_type);
}

kwiver::vital::config_block_keys_t
wrap_process
::_available_config() const
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_available_config");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f())
    }
  }

  return _base_available_config();
}

sprokit::process::conf_info_t
wrap_process
::_config_info(kwiver::vital::config_block_key_t const& key)
{
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_config_info");

    if (f)
    {
      SPROKIT_PYTHON_HANDLE_EXCEPTION(return f(key))
    }
  }

  return _base_config_info(key);
}

void
wrap_process
::_declare_input_port(port_t const& port, port_info_t const& info)
{
  declare_input_port(port, info);
}

void
wrap_process
::_declare_input_port_1(port_t const& port,
                        port_type_t const& type_,
                        port_flags_t const& flags_,
                        port_description_t const& description_,
                        port_frequency_t const& frequency_)
{
  declare_input_port(port, type_, flags_, description_, frequency_);
}

void
wrap_process
::_declare_output_port(port_t const& port, port_info_t const& info)
{
  declare_output_port(port, info);
}

void
wrap_process
::_declare_output_port_1(port_t const& port,
                         port_type_t const& type_,
                         port_flags_t const& flags_,
                         port_description_t const& description_,
                         port_frequency_t const& frequency_)
{
  declare_output_port(port, type_, flags_, description_, frequency_);
}

void
wrap_process
::_set_input_port_frequency(port_t const& port, port_frequency_t const& new_frequency)
{
  set_input_port_frequency(port, new_frequency);
}

void
wrap_process
::_set_output_port_frequency(port_t const& port, port_frequency_t const& new_frequency)
{
  set_output_port_frequency(port, new_frequency);
}

void
wrap_process
::_remove_input_port(port_t const& port)
{
  remove_input_port(port);
}

void
wrap_process
::_remove_output_port(port_t const& port)
{
  remove_output_port(port);
}

void
wrap_process
::_declare_configuration_key(kwiver::vital::config_block_key_t const& key, conf_info_t const& info)
{
  declare_configuration_key(key, info);
}

void
wrap_process
::_declare_configuration_key_1(kwiver::vital::config_block_key_t const& key,
                               kwiver::vital::config_block_value_t const& def_,
                               kwiver::vital::config_block_description_t const& description_)
{
  declare_configuration_key(key, def_, description_);
}

void
wrap_process
::_declare_configuration_key_2(kwiver::vital::config_block_key_t const& key,
                               kwiver::vital::config_block_value_t const& def_,
                               kwiver::vital::config_block_description_t const& description_,
                               bool tunable_)
{
  declare_configuration_key(key, def_, description_, tunable_);
}

void
wrap_process
::_mark_process_as_complete()
{
  mark_process_as_complete();
}

bool
wrap_process
::_has_input_port_edge(port_t const& port) const
{
  return has_input_port_edge(port);
}

size_t
wrap_process
::_count_output_port_edges(port_t const& port) const
{
  return count_output_port_edges(port);
}

sprokit::edge_datum_t
wrap_process
::_peek_at_port(port_t const& port, size_t idx) const
{
  return peek_at_port(port, idx);
}

sprokit::datum_t
wrap_process
::_peek_at_datum_on_port(port_t const& port, size_t idx) const
{
  return peek_at_datum_on_port(port, idx);
}

sprokit::edge_datum_t
wrap_process
::_grab_from_port(port_t const& port) const
{
  return grab_from_port(port);
}

sprokit::datum_t
wrap_process
::_grab_datum_from_port(port_t const& port) const
{
  return grab_datum_from_port(port);
}

object
wrap_process
::_grab_value_from_port(port_t const& port) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  sprokit::datum_t const dat = grab_datum_from_port(port);
  boost::any const any = dat->get_datum<boost::any>();

  return object(any);
}

void
wrap_process
::_push_to_port(port_t const& port, sprokit::edge_datum_t const& dat) const
{
  return push_to_port(port, dat);
}

void
wrap_process
::_push_datum_to_port(port_t const& port, sprokit::datum_t const& dat) const
{
  return push_datum_to_port(port, dat);
}

void
wrap_process
::_push_value_to_port(port_t const& port, object const& obj) const
{
  sprokit::python::python_gil const gil;

  (void)gil;

  boost::any const any = extract<boost::any>(obj)();
  sprokit::datum_t const dat = sprokit::datum::new_datum(any);

  return push_datum_to_port(port, dat);
}

kwiver::vital::config_block_sptr
wrap_process
::_get_config() const
{
  return get_config();
}

kwiver::vital::config_block_value_t
wrap_process
::_config_value(kwiver::vital::config_block_key_t const& key) const
{
  return config_value<kwiver::vital::config_block_value_t>(key);
}

void
wrap_process
::_set_data_checking_level(data_check_t check)
{
  set_data_checking_level(check);
}

sprokit::process::data_info_t
wrap_process
::_edge_data_info(sprokit::edge_data_t const& data)
{
  return edge_data_info(data);
}
