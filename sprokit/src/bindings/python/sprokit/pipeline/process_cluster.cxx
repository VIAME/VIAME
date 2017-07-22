/*ckwg +29
 * Copyright 2012-2017 by Kitware, Inc.
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

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>

#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/override.hpp>
#if WIN32
#pragma warning (pop)
#endif

/**
 * \file process_cluster.cxx
 *
 * \brief Python bindings for \link sprokit::process_cluster\endlink.
 */

using namespace boost::python;

class wrap_process_cluster
  : public sprokit::process_cluster
  , public wrapper<sprokit::process_cluster>
{
  public:
    wrap_process_cluster(kwiver::vital::config_block_sptr const& config);
    ~wrap_process_cluster();

    properties_t _base_properties() const;

    void _base_reconfigure(kwiver::vital::config_block_sptr const& conf);

    sprokit::processes_t processes() const;
    connections_t input_mappings() const;
    connections_t output_mappings() const;
    connections_t internal_connections() const;

    void _map_config(kwiver::vital::config_block_key_t const& key, name_t const& name_, kwiver::vital::config_block_key_t const& mapped_key);
    void _add_process(name_t const& name_, type_t const& type_, kwiver::vital::config_block_sptr const& config);
    void _map_input(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _map_output(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _connect(name_t const& upstream_name, port_t const& upstream_port,
                  name_t const& downstream_name, port_t const& downstream_port);

    properties_t _properties() const;

    void _reconfigure(kwiver::vital::config_block_sptr const& conf);

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

    void _declare_configuration_key(kwiver::vital::config_block_key_t const& key, conf_info_t const& info);
    void _declare_configuration_key_1(kwiver::vital::config_block_key_t const& key,
                                      kwiver::vital::config_block_value_t const& def_,
                                      kwiver::vital::config_block_description_t const& description_);
};

static object cluster_from_process(sprokit::process_t const& process);

BOOST_PYTHON_MODULE(process_cluster)
{
  class_<wrap_process_cluster, boost::noncopyable>("PythonProcessCluster"
    , "The base class for Python process clusters."
    , no_init)
    .def(init<kwiver::vital::config_block_sptr>())
    .def("name", &sprokit::process::name
      , "Returns the name of the process.")
    .def("type", &sprokit::process::type
      , "Returns the type of the process.")
    .def_readonly("property_no_threads", &sprokit::process::property_no_threads)
    .def_readonly("property_no_reentrancy", &sprokit::process::property_no_reentrancy)
    .def_readonly("property_unsync_input", &sprokit::process::property_unsync_input)
    .def_readonly("property_unsync_output", &sprokit::process::property_unsync_output)
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
    .def("_base_properties", &wrap_process_cluster::_base_properties
      , "Base class properties.")
    .def("_base_reconfigure", &wrap_process_cluster::_base_reconfigure
      , (arg("config"))
      , "Base class reconfigure.")
    .def("map_config", &wrap_process_cluster::_map_config
      , (arg("key"), arg("name"), arg("mapped_key"))
      , "Map a configuration value to a process.")
    .def("add_process", &wrap_process_cluster::_add_process
      , (arg("name"), arg("type"), arg("config") = kwiver::vital::config_block::empty_config())
      , "Add a process to the cluster.")
    .def("map_input", &wrap_process_cluster::_map_input
      , (arg("port"), arg("name"), arg("mapped_port"))
      , "Map a port on the cluster to an input port.")
    .def("map_output", &wrap_process_cluster::_map_output
      , (arg("port"), arg("name"), arg("mapped_port"))
      , "Map an output port to a port on the cluster.")
    .def("connect", &wrap_process_cluster::_connect
      , (arg("upstream_name"), arg("upstream_port"), arg("downstream_name"), arg("downstream_port"))
      , "Connect two ports within the cluster.")
    .def("_properties", &wrap_process_cluster::_properties, &wrap_process_cluster::_base_properties
      , "The properties on the subclass.")
    .def("_reconfigure", &wrap_process_cluster::_reconfigure, &wrap_process_cluster::_base_reconfigure
      , (arg("config"))
      , "Runtime configuration for subclasses.")
    .def("declare_input_port", &wrap_process_cluster::_declare_input_port
      , (arg("port"), arg("info"))
      , "Declare an input port on the process.")
    .def("declare_input_port", &wrap_process_cluster::_declare_input_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1))
      , "Declare an input port on the process.")
    .def("declare_output_port", &wrap_process_cluster::_declare_output_port
      , (arg("port"), arg("info"))
      , "Declare an output port on the process.")
    .def("declare_output_port", &wrap_process_cluster::_declare_output_port_1
      , (arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1))
      , "Declare an output port on the process.")
    .def("declare_configuration_key", &wrap_process_cluster::_declare_configuration_key
      , (arg("key"), arg("info"))
      , "Declare a configuration key for the process.")
    .def("declare_configuration_key", &wrap_process_cluster::_declare_configuration_key_1
      , (arg("key"), arg("default"), arg("description"))
      , "Declare a configuration key for the process.")
    .def("processes", &sprokit::process_cluster::processes
      , "Processes in the cluster.")
    .def("input_mappings", &sprokit::process_cluster::input_mappings
      , "Input mappings for the cluster.")
    .def("output_mappings", &sprokit::process_cluster::output_mappings
      , "Output mappings for the cluster.")
    .def("internal_connections", &sprokit::process_cluster::internal_connections
      , "Connections internal to the cluster.")
  ;

  def("cluster_from_process", cluster_from_process
    , (arg("process"))
    , "Returns the process as a cluster or None if the process is not a cluster.");

  implicitly_convertible<sprokit::process_cluster_t, sprokit::process_t>();
}

wrap_process_cluster
::wrap_process_cluster(kwiver::vital::config_block_sptr const& config)
  : sprokit::process_cluster(config)
{
}

wrap_process_cluster
::~wrap_process_cluster()
{
}

sprokit::process::properties_t
wrap_process_cluster
::_base_properties() const
{
  return process_cluster::_properties();
}

void
wrap_process_cluster
::_base_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  return process_cluster::_reconfigure(conf);
}

void
wrap_process_cluster
::_map_config(kwiver::vital::config_block_key_t const& key, name_t const& name_, kwiver::vital::config_block_key_t const& mapped_key)
{
  map_config(key, name_, mapped_key);
}

void
wrap_process_cluster
::_add_process(name_t const& name_, type_t const& type_, kwiver::vital::config_block_sptr const& conf)
{
  add_process(name_, type_, conf);
}

void
wrap_process_cluster
::_map_input(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  map_input(port, name_, mapped_port);
}

void
wrap_process_cluster
::_map_output(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  map_output(port, name_, mapped_port);
}

void
wrap_process_cluster
::_connect(name_t const& upstream_name, port_t const& upstream_port,
           name_t const& downstream_name, port_t const& downstream_port)
{
  connect(upstream_name, upstream_port,
          downstream_name, downstream_port);
}

sprokit::process::properties_t
wrap_process_cluster
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

void
wrap_process_cluster
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
::_declare_configuration_key(kwiver::vital::config_block_key_t const& key, conf_info_t const& info)
{
  declare_configuration_key(key, info);
}

void
wrap_process_cluster
::_declare_configuration_key_1(kwiver::vital::config_block_key_t const& key,
                               kwiver::vital::config_block_value_t const& def_,
                               kwiver::vital::config_block_description_t const& description_)
{
  declare_configuration_key(key, def_, description_);
}

object
cluster_from_process(sprokit::process_t const& process)
{
  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(process);

  if (!cluster)
  {
    return object();
  }

  return object(cluster);
}
