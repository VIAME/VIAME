/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_exceptions.h>

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_cluster.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/override.hpp>
#include <boost/python/pure_virtual.hpp>

/**
 * \file process_cluster.cxx
 *
 * \brief Python bindings for \link vistk::process_cluster\endlink.
 */

using namespace boost::python;

class wrap_process_cluster
  : public vistk::process_cluster
  , public wrapper<vistk::process_cluster>
{
  public:
    wrap_process_cluster(vistk::config_t const& config);
    ~wrap_process_cluster();

    properties_t _base_properties() const;

    vistk::processes_t processes() const;
    connections_t input_mappings() const;
    connections_t output_mappings() const;
    connections_t internal_connections() const;

    properties_t _properties() const;

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

    void _declare_configuration_key(vistk::config::key_t const& key, conf_info_t const& info);
    void _declare_configuration_key_1(vistk::config::key_t const& key,
                                      vistk::config::value_t const& def_,
                                      vistk::config::description_t const& description_);

    override get_pure_override(char const* name) const;
};

static object cluster_from_process(vistk::process_t const& process);

BOOST_PYTHON_MODULE(process_cluster)
{
  class_<wrap_process_cluster, boost::noncopyable>("PythonProcessCluster"
    , "The base class for Python process clusters."
    , no_init)
    .def(init<vistk::config_t>())
    .def("name", &vistk::process::name
      , "Returns the name of the process.")
    .def("type", &vistk::process::type
      , "Returns the type of the process.")
    .def_readonly("type_any", &vistk::process::type_any)
    .def_readonly("type_none", &vistk::process::type_none)
    .def_readonly("type_data_dependent", &vistk::process::type_data_dependent)
    .def_readonly("type_flow_dependent", &vistk::process::type_flow_dependent)
    .def_readonly("flag_output_const", &vistk::process::flag_output_const)
    .def_readonly("flag_input_static", &vistk::process::flag_input_static)
    .def_readonly("flag_input_mutable", &vistk::process::flag_input_mutable)
    .def_readonly("flag_input_nodep", &vistk::process::flag_input_nodep)
    .def_readonly("flag_required", &vistk::process::flag_required)
    .def("_base_properties", &wrap_process_cluster::_base_properties
      , "Base class properties.")
    .def("_properties", &wrap_process_cluster::_properties, &wrap_process_cluster::_base_properties
      , "The properties on the subclass.")
    .def("declare_input_port", &wrap_process_cluster::_declare_input_port
      , (arg("port"), arg("info"))
      , "Declare an input port on the process.")
    .def("declare_input_port", &wrap_process_cluster::_declare_input_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = vistk::process::port_frequency_t(1))
      , "Declare an input port on the process.")
    .def("declare_output_port", &wrap_process_cluster::_declare_output_port
      , (arg("port"), arg("info"))
      , "Declare an output port on the process.")
    .def("declare_output_port", &wrap_process_cluster::_declare_output_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = vistk::process::port_frequency_t(1))
      , "Declare an output port on the process.")
    .def("declare_configuration_key", &wrap_process_cluster::_declare_configuration_key
      , (arg("key"), arg("info"))
      , "Declare a configuration key for the process.")
    .def("declare_configuration_key", &wrap_process_cluster::_declare_configuration_key_1
      , (arg("key"), arg("default"), arg("description"))
      , "Declare a configuration key for the process.")
    .def("processes", pure_virtual(&wrap_process_cluster::processes)
      , "Processes in the cluster.")
    .def("input_mappings", pure_virtual(&wrap_process_cluster::input_mappings)
      , "Input mappings for the cluster.")
    .def("output_mappings", pure_virtual(&wrap_process_cluster::output_mappings)
      , "Output mappings for the cluster.")
    .def("internal_connections", pure_virtual(&wrap_process_cluster::internal_connections)
      , "Connections internal to the cluster.")
  ;

  def("cluster_from_process", cluster_from_process
    , (arg("process"))
    , "Returns the process as a cluster or None if the process is not a cluster.");
}

wrap_process_cluster
::wrap_process_cluster(vistk::config_t const& config)
  : vistk::process_cluster(config)
{
}

wrap_process_cluster
::~wrap_process_cluster()
{
}

vistk::process::properties_t
wrap_process_cluster
::_base_properties() const
{
  return process_cluster::properties();
}

vistk::processes_t
wrap_process_cluster
::processes() const
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(return get_pure_override("processes")())
}

vistk::process::connections_t
wrap_process_cluster
::input_mappings() const
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(return get_pure_override("input_mappings")())
}

vistk::process::connections_t
wrap_process_cluster
::output_mappings() const
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(return get_pure_override("output_mappings")())
}

vistk::process::connections_t
wrap_process_cluster
::internal_connections() const
{
  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION(return get_pure_override("internal_connections")())
}

vistk::process::properties_t
wrap_process_cluster
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

void
wrap_process_cluster
::_declare_input_port(port_t const& port, port_info_t const& info)
{
  declare_input_port(port, info);
}

void
wrap_process_cluster
::_declare_input_port_1(port_t const& port,
                        port_type_t const& type_,
                        port_flags_t const& flags_,
                        port_description_t const& description_,
                        port_frequency_t const& frequency_)
{
  declare_input_port(port, type_, flags_, description_, frequency_);
}

void
wrap_process_cluster
::_declare_output_port(port_t const& port, port_info_t const& info)
{
  declare_output_port(port, info);
}

void
wrap_process_cluster
::_declare_output_port_1(port_t const& port,
                         port_type_t const& type_,
                         port_flags_t const& flags_,
                         port_description_t const& description_,
                         port_frequency_t const& frequency_)
{
  declare_output_port(port, type_, flags_, description_, frequency_);
}

void
wrap_process_cluster
::_declare_configuration_key(vistk::config::key_t const& key, conf_info_t const& info)
{
  declare_configuration_key(key, info);
}

void
wrap_process_cluster
::_declare_configuration_key_1(vistk::config::key_t const& key,
                               vistk::config::value_t const& def_,
                               vistk::config::description_t const& description_)
{
  declare_configuration_key(key, def_, description_);
}

object
cluster_from_process(vistk::process_t const& process)
{
  vistk::process_cluster_t const cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(process);

  if (!cluster)
  {
    return object();
  }

  return object(cluster);
}

override
wrap_process_cluster
::get_pure_override(char const* method) const
{
  override const o = get_override(method);

  if (!o)
  {
    std::ostringstream sstr;

    sstr << method << " is not implemented";

    throw std::runtime_error(sstr.str().c_str());
  }

  return o;
}
