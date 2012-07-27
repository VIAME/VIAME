/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_exceptions.h>
#include <python/helpers/python_wrap_const_shared_ptr.h>
#include <python/helpers/set_indexing_suite.h>

#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <vistk/python/util/python_gil.h>

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
 * \brief Python bindings for \link vistk::process\endlink.
 */

using namespace boost::python;

static vistk::process::property_t const property_python = vistk::process::property_t("_python");

/// \todo How to do grab_input_as<>?

class wrap_process
  : public vistk::process
  , public wrapper<vistk::process>
{
  public:
    wrap_process(vistk::config_t const& config);
    ~wrap_process();

    void _base_configure();

    void _base_init();

    void _base_reset();

    void _base_flush();

    void _base_step();

    properties_t _base_properties() const;

    ports_t _base_input_ports() const;
    ports_t _base_output_ports() const;

    port_info_t _base_input_port_info(port_t const& port);
    port_info_t _base_output_port_info(port_t const& port);

    bool _base_set_input_port_type(port_t const& port, port_type_t const& new_type);
    bool _base_set_output_port_type(port_t const& port, port_type_t const& new_type);

    vistk::config::keys_t _base_available_config() const;

    conf_info_t _base_config_info(vistk::config::key_t const& key);

    void _configure();

    void _init();

    void _reset();

    void _flush();

    void _step();

    properties_t _properties() const;

    ports_t _input_ports() const;
    ports_t _output_ports() const;

    port_info_t _input_port_info(port_t const& port);
    port_info_t _output_port_info(port_t const& port);

    bool _set_input_port_type(port_t const& port, port_type_t const& new_type);
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type);

    vistk::config::keys_t _available_config() const;

    conf_info_t _config_info(vistk::config::key_t const& key);

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

    void _remove_input_port(port_t const& port);
    void _remove_output_port(port_t const& port);

    void _declare_configuration_key(vistk::config::key_t const& key, conf_info_t const& info);
    void _declare_configuration_key_1(vistk::config::key_t const& key,
                                      vistk::config::value_t const& def_,
                                      vistk::config::description_t const& description_);

    void _mark_process_as_complete();

    bool _has_input_port_edge(port_t const& port) const;
    size_t _count_output_port_edges(port_t const& port) const;

    vistk::edge_datum_t _grab_from_port(port_t const& port) const;
    vistk::datum_t _grab_datum_from_port(port_t const& port) const;
    object _grab_value_from_port(port_t const& port) const;
    void _push_to_port(port_t const& port, vistk::edge_datum_t const& dat) const;
    void _push_datum_to_port(port_t const& port, vistk::datum_t const& dat) const;
    void _push_value_to_port(port_t const& port, object const& obj) const;

    vistk::config_t _get_config() const;
    vistk::config::value_t _config_value(vistk::config::key_t const& key) const;

    void _set_data_checking_level(data_check_t check);

    data_info_t _edge_data_info(vistk::edge_data_t const& data);
};

BOOST_PYTHON_MODULE(process)
{
  class_<vistk::process::name_t>("ProcessName"
    , "A type for the name of a process.");
  class_<vistk::process::names_t>("ProcessNames"
    , "A collection of process names.")
    .def(vector_indexing_suite<vistk::process::names_t>())
  ;
  class_<vistk::process::type_t>("ProcessType"
    , "The type for a type of process.");
  class_<vistk::process::types_t>("ProcessTypes"
    , "A collection of process types.")
    .def(vector_indexing_suite<vistk::process::types_t>())
  ;
  class_<vistk::process::property_t>("ProcessProperty"
    , "A property on a process.");
  class_<vistk::process::properties_t>("ProcessProperties"
    , "A collection of properties on a process.")
    .def(set_indexing_suite<vistk::process::properties_t>())
  ;
  class_<vistk::process::port_description_t>("PortDescription"
    , "A description for a port.");
  class_<vistk::process::port_t>("Port"
    , "The name of a port.");
  class_<vistk::process::ports_t>("Ports"
    , "A collection of ports.")
    .def(vector_indexing_suite<vistk::process::ports_t>())
  ;
  class_<vistk::process::port_type_t>("PortType"
    , "The type of data on a port.");
  class_<vistk::process::port_flag_t>("PortFlag"
    , "A flag on a port.");
  class_<vistk::process::port_flags_t>("PortFlags"
    , "A collection of port flags.")
    .def(set_indexing_suite<vistk::process::port_flags_t>())
  ;
  class_<vistk::process::port_frequency_t>("PortFrequency"
    , "A frequency for a port."
    , no_init)
    .def(init<vistk::process::frequency_component_t>())
    .def(init<vistk::process::frequency_component_t, vistk::process::frequency_component_t>())
    .def("numerator", &vistk::process::port_frequency_t::numerator
      , "The numerator of the frequency.")
    .def("denominator", &vistk::process::port_frequency_t::denominator
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
  class_<vistk::process::port_addr_t>("PortAddr"
    , "An address for a port within a pipeline.")
    .def_readwrite("process", &vistk::process::port_addr_t::first)
    .def_readwrite("port", &vistk::process::port_addr_t::second)
  ;
  class_<vistk::process::port_addrs_t>("PortAddrs"
    , "A collection of port addresses.")
    .def(vector_indexing_suite<vistk::process::port_addrs_t>())
  ;
  class_<vistk::process::connection_t>("Connection"
    , "A connection between two ports.")
    .def_readwrite("upstream", &vistk::process::connection_t::first)
    .def_readwrite("downstream", &vistk::process::connection_t::second)
  ;
  class_<vistk::process::connections_t>("Connections"
    , "A collection of connections.")
    .def(vector_indexing_suite<vistk::process::connections_t>())
  ;

  class_<vistk::process::port_info, vistk::process::port_info_t>("PortInfo"
    , "Information about a port on a process."
    , no_init)
    .def(init<vistk::process::port_type_t, vistk::process::port_flags_t, vistk::process::port_description_t, vistk::process::port_frequency_t>())
    .def_readonly("type", &vistk::process::port_info::type)
    .def_readonly("flags", &vistk::process::port_info::flags)
    .def_readonly("description", &vistk::process::port_info::description)
    .def_readonly("frequency", &vistk::process::port_info::frequency)
  ;

  implicitly_convertible<boost::shared_ptr<vistk::process::port_info>, vistk::process::port_info_t>();

  class_<vistk::process::conf_info, vistk::process::conf_info_t>("ConfInfo"
    , "Information about a configuration on a process."
    , no_init)
    .def(init<vistk::config::value_t, vistk::config::description_t>())
    .def_readonly("default", &vistk::process::conf_info::def)
    .def_readonly("description", &vistk::process::conf_info::description)
  ;

  implicitly_convertible<boost::shared_ptr<vistk::process::conf_info>, vistk::process::conf_info_t>();

  class_<vistk::process::data_info, vistk::process::data_info_t>("DataInfo"
    , "Information about a set of data packets from edges."
    , no_init)
    .def(init<bool, vistk::datum::type_t>())
    .def_readonly("in_sync", &vistk::process::data_info::in_sync)
    .def_readonly("max_status", &vistk::process::data_info::max_status)
  ;

  enum_<vistk::process::data_check_t>("DataCheck"
    , "Levels of input validation")
    .value("none", vistk::process::check_none)
    .value("sync", vistk::process::check_sync)
    .value("valid", vistk::process::check_valid)
  ;

  implicitly_convertible<boost::shared_ptr<vistk::process::data_info>, vistk::process::data_info_t>();

  class_<wrap_process, boost::noncopyable>("PythonProcess"
    , "The base class for Python processes."
    , no_init)
    .def(init<vistk::config_t>())
    .def("configure", &vistk::process::configure
      , "Configure the process.")
    .def("init", &vistk::process::init
      , "Initializes the process.")
    .def("reset", &vistk::process::reset
      , "Resets the process.")
    .def("step", &vistk::process::step
      , "Steps the process for one iteration.")
    .def("properties", &vistk::process::properties
      , "Returns the properties on the process.")
    .def("connect_input_port", &vistk::process::connect_input_port
      , (arg("port"), arg("edge"))
      , "Connects the given edge to the input port.")
    .def("connect_output_port", &vistk::process::connect_output_port
      , (arg("port"), arg("edge"))
      , "Connects the given edge to the output port.")
    .def("input_ports", &vistk::process::input_ports
      , "Returns a list of input ports on the process.")
    .def("output_ports", &vistk::process::output_ports
      , "Returns a list of output ports on the process.")
    .def("input_port_info", &vistk::process::input_port_info
      , (arg("port"))
      , "Returns information about the given input port.")
    .def("output_port_info", &vistk::process::output_port_info
      , (arg("port"))
      , "Returns information about the given output port.")
    .def("set_input_port_type", &vistk::process::set_input_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an input port.")
    .def("set_output_port_type", &vistk::process::set_output_port_type
      , (arg("port"), arg("new_type"))
      , "Sets the type for an output port.")
    .def("available_config", &vistk::process::available_config
      , "Returns a list of available configuration keys for the process.")
    .def("config_info", &vistk::process::config_info
      , (arg("config"))
      , "Returns information about the given configuration key.")
    .def("name", &vistk::process::name
      , "Returns the name of the process.")
    .def("type", &vistk::process::type
      , "Returns the type of the process.")
    .def_readonly("property_no_threads", &vistk::process::property_no_threads)
    .def_readonly("property_no_reentrancy", &vistk::process::property_no_reentrancy)
    .def_readonly("property_unsync_input", &vistk::process::property_unsync_input)
    .def_readonly("property_unsync_output", &vistk::process::property_unsync_output)
    .def_readonly("port_heartbeat", &vistk::process::port_heartbeat)
    .def_readonly("config_name", &vistk::process::config_name)
    .def_readonly("config_type", &vistk::process::config_type)
    .def_readonly("type_any", &vistk::process::type_any)
    .def_readonly("type_none", &vistk::process::type_none)
    .def_readonly("type_data_dependent", &vistk::process::type_data_dependent)
    .def_readonly("type_flow_dependent", &vistk::process::type_flow_dependent)
    .def_readonly("flag_output_const", &vistk::process::flag_output_const)
    .def_readonly("flag_input_static", &vistk::process::flag_input_static)
    .def_readonly("flag_input_mutable", &vistk::process::flag_input_mutable)
    .def_readonly("flag_input_nodep", &vistk::process::flag_input_nodep)
    .def_readonly("flag_required", &vistk::process::flag_required)
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
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = vistk::process::port_frequency_t(1))
      , "Declare an input port on the process.")
    .def("declare_output_port", &wrap_process::_declare_output_port
      , (arg("port"), arg("info"))
      , "Declare an output port on the process.")
    .def("declare_output_port", &wrap_process::_declare_output_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = vistk::process::port_frequency_t(1))
      , "Declare an output port on the process.")
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
    .def("mark_process_as_complete", &wrap_process::_mark_process_as_complete
      , "Tags the process as complete.")
    .def("has_input_port_edge", &wrap_process::_has_input_port_edge
      , (arg("port"))
      , "True if there is an edge that is connected to the port, False otherwise.")
    .def("count_output_port_edges", &wrap_process::_count_output_port_edges
      , (arg("port"))
      , "The number of edges that are connected to the port.")
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
::wrap_process(vistk::config_t const& config)
  : vistk::process(config)
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
  TRANSLATE_PYTHON_EXCEPTION(process::_configure())
}

void
wrap_process
::_base_init()
{
  TRANSLATE_PYTHON_EXCEPTION(process::_init())
}

void
wrap_process
::_base_reset()
{
  TRANSLATE_PYTHON_EXCEPTION(process::_reset())
}

void
wrap_process
::_base_flush()
{
  TRANSLATE_PYTHON_EXCEPTION(process::_flush())
}

void
wrap_process
::_base_step()
{
  TRANSLATE_PYTHON_EXCEPTION(process::_step())
}

vistk::process::properties_t
wrap_process
::_base_properties() const
{
  properties_t consts = process::_properties();

  consts.insert(property_python);

  return consts;
}

vistk::process::ports_t
wrap_process
::_base_input_ports() const
{
  return process::_input_ports();
}

vistk::process::ports_t
wrap_process
::_base_output_ports() const
{
  return process::_output_ports();
}

vistk::process::port_info_t
wrap_process
::_base_input_port_info(port_t const& port)
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_input_port_info(port))
}

vistk::process::port_info_t
wrap_process
::_base_output_port_info(port_t const& port)
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_output_port_info(port))
}

bool
wrap_process
::_base_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_set_input_port_type(port, new_type))
}

bool
wrap_process
::_base_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_set_input_port_type(port, new_type))
}

vistk::config::keys_t
wrap_process
::_base_available_config() const
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_available_config())
}

vistk::process::conf_info_t
wrap_process
::_base_config_info(vistk::config::key_t const& key)
{
  TRANSLATE_PYTHON_EXCEPTION(return process::_config_info(key))
}

void
wrap_process
::_configure()
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_configure");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(f())

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
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_init");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(f())

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
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_reset");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(f())

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
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_flush");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(f())

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
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_step");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(f())

      return;
    }
  }

  _base_step();
}

vistk::process::properties_t
wrap_process
::_properties() const
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_properties");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f())
    }
  }

  return _base_properties();
}

vistk::process::ports_t
wrap_process
::_input_ports() const
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_input_ports");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f())
    }
  }

  return _base_input_ports();
}

vistk::process::ports_t
wrap_process
::_output_ports() const
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_output_ports");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f())
    }
  }

  return _base_output_ports();
}

vistk::process::port_info_t
wrap_process
::_input_port_info(port_t const& port)
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_input_port_info");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f(port))
    }
  }

  return _base_input_port_info(port);
}

vistk::process::port_info_t
wrap_process
::_output_port_info(port_t const& port)
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_output_port_info");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f(port))
    }
  }

  return _base_output_port_info(port);
}

bool
wrap_process
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_set_input_port_type");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f(port, new_type))
    }
  }

  return _base_set_input_port_type(port, new_type);
}

bool
wrap_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_set_output_port_type");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f(port, new_type))
    }
  }

  return _base_set_output_port_type(port, new_type);
}

vistk::config::keys_t
wrap_process
::_available_config() const
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_available_config");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f())
    }
  }

  return _base_available_config();
}

vistk::process::conf_info_t
wrap_process
::_config_info(vistk::config::key_t const& key)
{
  {
    vistk::python::python_gil const gil;

    (void)gil;

    override const f = get_override("_config_info");

    if (f)
    {
      HANDLE_PYTHON_EXCEPTION(return f(key))
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
::_declare_configuration_key(vistk::config::key_t const& key, conf_info_t const& info)
{
  declare_configuration_key(key, info);
}

void
wrap_process
::_declare_configuration_key_1(vistk::config::key_t const& key,
                               vistk::config::value_t const& def_,
                               vistk::config::description_t const& description_)
{
  declare_configuration_key(key, def_, description_);
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

vistk::edge_datum_t
wrap_process
::_grab_from_port(port_t const& port) const
{
  return grab_from_port(port);
}

vistk::datum_t
wrap_process
::_grab_datum_from_port(port_t const& port) const
{
  return grab_datum_from_port(port);
}

object
wrap_process
::_grab_value_from_port(port_t const& port) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  vistk::datum_t const dat = grab_datum_from_port(port);
  boost::any const any = dat->get_datum<boost::any>();

  return object(any);
}

void
wrap_process
::_push_to_port(port_t const& port, vistk::edge_datum_t const& dat) const
{
  return push_to_port(port, dat);
}

void
wrap_process
::_push_datum_to_port(port_t const& port, vistk::datum_t const& dat) const
{
  return push_datum_to_port(port, dat);
}

void
wrap_process
::_push_value_to_port(port_t const& port, object const& obj) const
{
  vistk::python::python_gil const gil;

  (void)gil;

  boost::any const any = extract<boost::any>(obj)();
  vistk::datum_t const dat = vistk::datum::new_datum(any);

  return push_datum_to_port(port, dat);
}

vistk::config_t
wrap_process
::_get_config() const
{
  return get_config();
}

vistk::config::value_t
wrap_process
::_config_value(vistk::config::key_t const& key) const
{
  return config_value<vistk::config::value_t>(key);
}

void
wrap_process
::_set_data_checking_level(data_check_t check)
{
  set_data_checking_level(check);
}

vistk::process::data_info_t
wrap_process
::_edge_data_info(vistk::edge_data_t const& data)
{
  return edge_data_info(data);
}
