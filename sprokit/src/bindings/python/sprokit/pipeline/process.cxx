/*ckwg +29
 * Copyright 2011-2013, 2019 by Kitware, Inc.
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

#include <pybind11/stl.h>
typedef std::set<std::string> string_set; // This has to be done first thing, or the macro breaks
PYBIND11_MAKE_OPAQUE(string_set)

#include <vital/bindings/python/vital/util/pybind11.h>
#include <vital/bindings/python/vital/util/python_exceptions.h>

#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#include <utility>

#include "python_wrappers.cxx"

/**
 * \file process.cxx
 *
 * \brief Python bindings for \link sprokit::process\endlink.
 */

using namespace pybind11;

// Publicist class to access protected methods
class wrap_process
  : public sprokit::process
{
  public:
    using process::process;

    using process::_configure;
    using process::_init;
    using process::_reset;
    using process::_flush;
    using process::_step;
    using process::_reconfigure;
    using process::_properties;
    using process::_input_ports;
    using process::_output_ports;
    using process::_input_port_info;
    using process::_output_port_info;
    using process::_set_input_port_type;
    using process::_set_output_port_type;
    using process::_available_config;
    using process::_config_info;
    using process::declare_input_port;
    using process::declare_output_port;
    using process::declare_configuration_key;
    using process::set_input_port_frequency;
    using process::set_output_port_frequency;
    using process::remove_input_port;
    using process::remove_output_port;
    using process::mark_process_as_complete;
    using process::has_input_port_edge;
    using process::count_output_port_edges;
    using process::peek_at_port;
    using process::peek_at_datum_on_port;
    using process::grab_from_port;
    using process::grab_datum_from_port;
    using process::push_to_port;
    using process::push_datum_to_port;
    using process::get_config;
    using process::config_value;
    using process::set_data_checking_level;
};

// Trampoline class to allow us to use virtual methods
class process_trampoline
  : public sprokit::process
{
  public:
    using process::process;

    void _configure() override;
    void _init() override;
    void _reset() override;
    void _flush() override;
    void _step() override;
    void _reconfigure(kwiver::vital::config_block_sptr const& config) override;
    sprokit::process::properties_t _properties() const override;
    sprokit::process::properties_t _properties_over() const;
    sprokit::process::ports_t _input_ports() const override;
    sprokit::process::ports_t _output_ports() const override;
    port_info_t _input_port_info(port_t const& port) override;
    port_info_t _output_port_info(port_t const& port) override;
    bool _set_input_port_type(port_t const& port, port_type_t const& new_type) override;
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type) override;
    kwiver::vital::config_block_keys_t _available_config() const override;
    sprokit::process::conf_info_t _config_info(kwiver::vital::config_block_key_t const& key) override;

};

void declare_input_port_2(sprokit::process &self, sprokit::process::port_t const& port, sprokit::process::port_info_t const& port_info);
void declare_input_port_5(sprokit::process &self,
                          sprokit::process::port_t const& port,
                          sprokit::process::port_type_t const& type_,
                          sprokit::process::port_flags_t const& flags_,
                          sprokit::process::port_description_t const& description_,
                          sprokit::process::port_frequency_t const& frequency_);

void declare_output_port_2(sprokit::process &self, sprokit::process::port_t const& port, sprokit::process::port_info_t const& port_info);
void declare_output_port_5(sprokit::process &self,
                          sprokit::process::port_t const& port,
                          sprokit::process::port_type_t const& type_,
                          sprokit::process::port_flags_t const& flags_,
                          sprokit::process::port_description_t const& description_,
                          sprokit::process::port_frequency_t const& frequency_);

void declare_configuration_key_2(sprokit::process &self,
                                 kwiver::vital::config_block_key_t const& key,
                                 sprokit::process::conf_info_t const& info);
void declare_configuration_key_3(sprokit::process &self,
                                 kwiver::vital::config_block_key_t const& key,
                                 kwiver::vital::config_block_value_t const& def_,
                                 kwiver::vital::config_block_description_t const& description_);
void declare_configuration_key_4(sprokit::process &self,
                                 kwiver::vital::config_block_key_t const& key,
                                 kwiver::vital::config_block_value_t const& def_,
                                 kwiver::vital::config_block_description_t const& description_,
                                 bool tunable_);

wrap_edge_datum peek_at_port(sprokit::process &self, sprokit::process::port_t const& port, std::size_t idx);
wrap_edge_datum grab_from_port(sprokit::process &self, sprokit::process::port_t const& port);
sprokit::datum grab_datum_from_port(sprokit::process &self, sprokit::process::port_t const& port);
sprokit::datum peek_at_datum_on_port(sprokit::process &self,
                                     sprokit::process::port_t const& port,
                                     std::size_t idx);
object grab_value_from_port(sprokit::process &self, sprokit::process::port_t const& port);

void push_value_to_port(sprokit::process &self, sprokit::process::port_t const& port, object const& obj);
void push_datum_to_port(sprokit::process &self, sprokit::process::port_t const& port, sprokit::datum const& dat);

std::string config_value(sprokit::process &self, kwiver::vital::config_block_key_t const& key);

PYBIND11_MODULE(process, m)
{
  bind_vector<sprokit::process::names_t>(m, "ProcessNames"
    , module_local()
    , "A collection of process names.")
  ;

  m.attr("ProcessTypes") = m.attr("ProcessNames");
  m.attr("ProcessTypes").attr("__main__") = "A collection of process types.";

  class_<string_set>(m, "ProcessProperties"
    , "A collection of properties on a process.")
    .def(init<>())
    .def("add", [](string_set &s, std::string item){ s.insert(item); })
    .def("pop", [](string_set &s) { if(s.empty())
                                    {
                                      throw key_error(".pop() on an empty set");
                                    }
                                    s.erase(--s.end()); })
    .def("remove", [](string_set &s, std::string item)
                                 { for ( auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if(*itr == item)
                                     {
                                       s.erase(itr);
                                       return;
                                     }
                                   }
                                   throw key_error(".remove() with an item that does not exist in the set");
                                 })
    .def("discard", [](string_set &s, std::string item)
                                 { for ( auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if(*itr == item)
                                     {
                                       s.erase(itr);
                                       return;
                                     }
                                   }
                                   return;
                                 })
    .def("isdisjoint", [](string_set &s, string_set &other)
                                 { for ( auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if( other.find(*itr) != other.end() )
                                     {
                                       return false;
                                     }
                                   }
                                   return true;
                                 })
    .def("issubset", [](string_set &s, string_set &other)
                                 { for (auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if( other.find(*itr) == other.end() )
                                     {
                                       return false;
                                     }
                                   }
                                   return true;
                                 })
    .def("issuperset", [](string_set &s, string_set &other)
                                 { for (auto itr = other.begin(); itr != other.end(); itr++)
                                   {
                                     if( s.find(*itr) == s.end() )
                                     {
                                       return false;
                                     }
                                   }
                                   return true;
                                 })
    .def("union", [](string_set &s, string_set &other)
                                 { string_set u(s);
                                   for (auto itr = other.begin(); itr != other.end(); itr++)
                                   {
                                     u.insert(*itr);
                                   }
                                   return u;
                                 })
    .def("difference", [](string_set &s, string_set &other)
                                 { string_set d;
                                   for (auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if( other.find(*itr) == other.end())
                                     {
                                       d.insert(*itr);
                                     }
                                   }
                                   return d;
                                 })
    .def("intersection", [](string_set &s, string_set &other)
                                 { string_set i;
                                   for (auto itr = s.begin(); itr != s.end(); itr++)
                                   {
                                     if( other.find(*itr) != other.end())
                                     {
                                       i.insert(*itr);
                                     }
                                   }
                                   return i;
                                 })
    .def("symmetric_difference", [](string_set &s, string_set &other)
                                 { string_set d(s);
                                   for (auto itr = other.begin(); itr != other.end(); itr++)
                                   {
                                     if( d.find(*itr) == d.end())
                                     {
                                       d.insert(*itr);
                                     }
                                     else
                                     {
                                       d.erase(d.find(*itr));
                                     }
                                   }
                                   return d;
                                 })
    .def("update", [](string_set &s, string_set &other)
                                 { for (auto itr = other.begin(); itr != other.end(); itr++)
                                   {
                                     s.insert(*itr);
                                   }
                                 })
    .def("copy", [](string_set &s) { return string_set(s); })
    .def("clear", &string_set::clear)
    .def("__len__", &string_set::size)
    .def("__contains__", [](string_set &s, std::string item){ return s.find(item) != s.end(); })
    .def("__iter__", [](string_set &s){return make_iterator(s.begin(),s.end());}, keep_alive<0,1>())
    ;

  m.attr("PortFlags") = m.attr("ProcessProperties");

  m.attr("Ports") = m.attr("ProcessNames");
  m.attr("Ports").attr("__doc__") = "A collection of ports.";

  class_<sprokit::process::port_frequency_t>(m, "PortFrequency"
    , "A frequency for a port.")
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
  class_<wrap_port_addr>(m, "PortAddr"
    , "An address for a port within a pipeline.")
    .def(init<>())
    .def_readwrite("process", &wrap_port_addr::process)
    .def_readwrite("port", &wrap_port_addr::port)
    .def("getAddr", &wrap_port_addr::get_addr)
  ;
  bind_vector<std::vector<wrap_port_addr> >(m, "PortAddrs"
    , "A collection of port addresses.")
  ;
  class_<sprokit::process::connection_t>(m, "Connection"
    , "A connection between two ports.")
    .def(init<>());
  ;
  bind_vector<sprokit::process::connections_t >(m, "Connections"
    , "A collection of connections.")
  ;

  class_<sprokit::process::port_info>(m, "PortInfo"
    , "Information about a port on a process.")
    .def(init<sprokit::process::port_type_t, std::set<std::string>, sprokit::process::port_description_t, sprokit::process::port_frequency_t>())
    .def_readonly("type", &sprokit::process::port_info::type)
    .def_readonly("flags", &sprokit::process::port_info::flags)
    .def_readonly("description", &sprokit::process::port_info::description)
    .def_readonly("frequency", &sprokit::process::port_info::frequency)
  ;

  class_<sprokit::process::conf_info, std::shared_ptr<sprokit::process::conf_info> >(m, "ConfInfo"
    , "Information about a configuration on a process.")
    .def(init<kwiver::vital::config_block_value_t, kwiver::vital::config_block_description_t, bool>())
    .def_readonly("default", &sprokit::process::conf_info::def)
    .def_readonly("description", &sprokit::process::conf_info::description)
    .def_readonly("tunable", &sprokit::process::conf_info::tunable)
  ;

  class_<sprokit::process::data_info>(m, "DataInfo"
    , "Information about a set of data packets from edges.")
    .def(init<bool, sprokit::datum::type_t>())
    .def_readonly("in_sync", &sprokit::process::data_info::in_sync)
    .def_readonly("max_status", &sprokit::process::data_info::max_status)
  ;

  enum_<sprokit::process::data_check_t>(m, "DataCheck"
    , "Levels of input validation")
    .value("none", sprokit::process::check_none)
    .value("sync", sprokit::process::check_sync)
    .value("valid", sprokit::process::check_valid)
  ;

  class_<sprokit::process, process_trampoline, sprokit::process_t>(m, "PythonProcess"
    , "The base class for Python processes.")
    .def(init<kwiver::vital::config_block_sptr>(), call_guard<kwiver::vital::python::gil_scoped_release>())
    .def("configure", &sprokit::process::configure, "Configure the process.", call_guard<kwiver::vital::python::gil_scoped_release>())
    .def("init", &sprokit::process::init, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Initializes the process.")
    .def("reset", &sprokit::process::reset, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Resets the process.")
    .def("step", &sprokit::process::step, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Steps the process for one iteration.")
    .def("properties", &sprokit::process::properties, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns the properties on the process.")
    .def("connect_input_port", &sprokit::process::connect_input_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("edge")
      , "Connects the given edge to the input port.")
    .def("connect_output_port", &sprokit::process::connect_output_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("edge")
      , "Connects the given edge to the output port.")
    .def("input_ports", &sprokit::process::input_ports, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns a list of input ports on the process.")
    .def("output_ports", &sprokit::process::output_ports, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns a list of output ports on the process.")
    .def("input_port_info", &sprokit::process::input_port_info, call_guard<kwiver::vital::python::gil_scoped_release>()
      , (arg("port"))
      , "Returns information about the given input port.")
    .def("output_port_info", &sprokit::process::output_port_info, call_guard<kwiver::vital::python::gil_scoped_release>()
      , (arg("port"))
      , "Returns information about the given output port.")
    .def("set_input_port_type", &sprokit::process::set_input_port_type, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("new_type")
      , "Sets the type for an input port.")
    .def("set_output_port_type", &sprokit::process::set_output_port_type, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("new_type")
      , "Sets the type for an output port.")
    .def("available_config", &sprokit::process::available_config, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns a list of available configuration keys for the process.")
    .def("available_tunable_config", &sprokit::process::available_tunable_config, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns a list of available tunable configuration keys for the process.")
    .def("config_info", &sprokit::process::config_info, call_guard<kwiver::vital::python::gil_scoped_release>()
      , (arg("config"))
      , "Returns information about the given configuration key.")

    .def("config_diff", &sprokit::process::config_diff, call_guard<kwiver::vital::python::gil_scoped_release>()
         , "Returns config difference information.")

    .def("name", &sprokit::process::name, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns the name of the process.")
    .def("type", &sprokit::process::type, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Returns the type of the process.")

    .def_readonly_static("property_no_threads", &sprokit::process::property_no_threads)
    .def_readonly_static("property_no_reentrancy", &sprokit::process::property_no_reentrancy)
    .def_readonly_static("property_unsync_input", &sprokit::process::property_unsync_input)
    .def_readonly_static("property_unsync_output", &sprokit::process::property_unsync_output)
    .def_readonly_static("port_heartbeat", &sprokit::process::port_heartbeat)
    .def_readonly_static("config_name", &sprokit::process::config_name)
    .def_readonly_static("config_type", &sprokit::process::config_type)
    .def_readonly_static("type_any", &sprokit::process::type_any)
    .def_readonly_static("type_none", &sprokit::process::type_none)
    .def_readonly_static("type_data_dependent", &sprokit::process::type_data_dependent)
    .def_readonly_static("type_flow_dependent", &sprokit::process::type_flow_dependent)
    .def_readonly_static("flag_output_const", &sprokit::process::flag_output_const)
    .def_readonly_static("flag_output_shared", &sprokit::process::flag_output_shared)
    .def_readonly_static("flag_input_static", &sprokit::process::flag_input_static)
    .def_readonly_static("flag_input_mutable", &sprokit::process::flag_input_mutable)
    .def_readonly_static("flag_input_nodep", &sprokit::process::flag_input_nodep)
    .def_readonly_static("flag_required", &sprokit::process::flag_required)
    .def("_base_configure", static_cast<void (sprokit::process::*)()>(&wrap_process::_configure), call_guard<kwiver::vital::python::gil_scoped_release>(), "Configure the base process.")
    .def("_base_init", static_cast<void (sprokit::process::*)()>(&wrap_process::_init), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class initialization.")
    .def("_base_reset", static_cast<void (sprokit::process::*)()>(&wrap_process::_reset), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class reset.")
    .def("_base_flush", static_cast<void (sprokit::process::*)()>(&wrap_process::_flush), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class flush.")
    .def("_base_step", static_cast<void (sprokit::process::*)()>(&wrap_process::_step), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class step.")
    .def("_base_reconfigure", static_cast<void (sprokit::process::*)(kwiver::vital::config_block_sptr const&)>(&wrap_process::_reconfigure), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("conf"), "Base class reconfigure.")
    .def("_base_properties", static_cast<sprokit::process::properties_t (sprokit::process::*)() const>(&wrap_process::_properties), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class properties.")
    .def("_base_input_ports", static_cast<sprokit::process::ports_t (sprokit::process::*)() const>(&wrap_process::_input_ports), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class input ports.")
    .def("_base_output_ports", static_cast<sprokit::process::ports_t (sprokit::process::*)() const>(&wrap_process::_output_ports), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class output ports.")
    .def("_base_input_port_info", static_cast<sprokit::process::port_info_t (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::_input_port_info), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), "Base class input port info.")
    .def("_base_output_port_info", static_cast<sprokit::process::port_info_t (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::_output_port_info), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), "Base class output port info.")
    .def("_base_set_input_port_type", static_cast<bool (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_type_t const&)>(&wrap_process::_set_input_port_type), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), arg("new_type"), "Base class input port type setting.")
    .def("_base_set_output_port_type", static_cast<bool (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_type_t const&)>(&wrap_process::_set_output_port_type), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), arg("new_type"), "Base class output port type setting.")
    .def("_base_available_config", static_cast<kwiver::vital::config_block_keys_t (sprokit::process::*)() const>(&wrap_process::_available_config), call_guard<kwiver::vital::python::gil_scoped_release>(), "Base class available configuration information.")
    .def("_base_config_info", static_cast<sprokit::process::conf_info_t (sprokit::process::*)(kwiver::vital::config_block_key_t const&)>(&wrap_process::_config_info), call_guard<kwiver::vital::python::gil_scoped_release>(), return_value_policy::reference, arg("config"), "Base class configuration information.")
    .def("_configure", static_cast<void (sprokit::process::*)()>(&wrap_process::_configure), call_guard<kwiver::vital::python::gil_scoped_release>(), "Configure the sub process.")
    .def("_init", static_cast<void (sprokit::process::*)()>(&wrap_process::_init), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class initialization.")
    .def("_reset", static_cast<void (sprokit::process::*)()>(&wrap_process::_reset), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class reset.")
    .def("_flush", static_cast<void (sprokit::process::*)()>(&wrap_process::_flush), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class flush.")
    .def("_step", static_cast<void (sprokit::process::*)()>(&wrap_process::_step), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class step.")
    .def("_reconfigure", static_cast<void (sprokit::process::*)(kwiver::vital::config_block_sptr const&)>(&wrap_process::_reconfigure), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("conf"), "Sub class reconfigure.")
    .def("_properties", static_cast<sprokit::process::properties_t (sprokit::process::*)() const>(&wrap_process::_properties), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class properties.")
    .def("_input_ports", static_cast<sprokit::process::ports_t (sprokit::process::*)() const>(&wrap_process::_input_ports), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class input ports.")
    .def("_output_ports", static_cast<sprokit::process::ports_t (sprokit::process::*)() const>(&wrap_process::_output_ports), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class output ports.")
    .def("_input_port_info", static_cast<sprokit::process::port_info_t (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::_input_port_info), arg("port"), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class input port info.")
    .def("_output_port_info", static_cast<sprokit::process::port_info_t (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::_output_port_info), arg("port"), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class output port info.")
    .def("_set_input_port_type", static_cast<bool (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_type_t const&)>(&wrap_process::_set_input_port_type), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), arg("new_type"), "Sub class input port type setting.")
    .def("_set_output_port_type", static_cast<bool (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_type_t const&)>(&wrap_process::_set_output_port_type), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("port"), arg("new_type"), "Sub class output port type setting.")
    .def("_available_config", static_cast<kwiver::vital::config_block_keys_t (sprokit::process::*)() const>(&wrap_process::_available_config), call_guard<kwiver::vital::python::gil_scoped_release>(), "Sub class available configuration information.")
    .def("_config_info", static_cast<sprokit::process::conf_info_t (sprokit::process::*)(kwiver::vital::config_block_key_t const&)>(&wrap_process::_config_info), call_guard<kwiver::vital::python::gil_scoped_release>(), arg("config"), "Sub class configuration information.")
    .def("declare_input_port", &declare_input_port_2, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("info")
      , "Declare an input port on the process.")
    .def("declare_input_port", &declare_input_port_5, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1)
      , "Declare an input port on the process.")
    .def("declare_output_port", &declare_output_port_2, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("info")
      , "Declare an output port on the process.")
    .def("declare_output_port", &declare_output_port_5, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("type"), arg("flags"), arg("description"), arg("frequency") = sprokit::process::port_frequency_t(1)
      , "Declare an output port on the process.")
    .def("declare_configuration_key", &declare_configuration_key_2, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("key"), arg("info")
      , "Declare a configuration key for the process")
    .def("declare_configuration_key", &declare_configuration_key_3, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("key"), arg("default"), arg("description")
      , "Declare a configuration key for the process")
    .def("declare_configuration_key", &declare_configuration_key_4, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("key"), arg("default"), arg("description"), arg("tunable")
      , "Declare a configuration key for the process")
    .def("set_input_port_frequency", static_cast<void (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_frequency_t const&)>(&wrap_process::set_input_port_frequency), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("new_frequency")
      , "Set an input port\'s frequency.")
    .def("set_output_port_frequency", static_cast<void (sprokit::process::*)(sprokit::process::port_t const&, sprokit::process::port_frequency_t const&)>(&wrap_process::set_output_port_frequency), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("new_frequency")
      , "Set an output port\'s frequency.")
    .def("remove_input_port", static_cast<void (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::remove_input_port), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "Remove an input port from the process.")
    .def("remove_output_port", static_cast<void (sprokit::process::*)(sprokit::process::port_t const&)>(&wrap_process::remove_output_port), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "Remove an output port from the process.")
    .def("mark_process_as_complete", static_cast<void (sprokit::process::*)()>(&wrap_process::mark_process_as_complete), call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Tags the process as complete.")
    .def("has_input_port_edge", static_cast<bool (sprokit::process::*)(sprokit::process::port_t const&) const>(&wrap_process::has_input_port_edge), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "True if there is an edge that is connected to the port, False otherwise.")
    .def("count_output_port_edges", static_cast<pybind11::size_t (sprokit::process::*)(sprokit::process::port_t const&) const>(&wrap_process::count_output_port_edges), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "The number of edges that are connected to a port.")
    .def("peek_at_port", &peek_at_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("idx") = 0
      , "Peek at a port.")
    .def("peek_at_datum_on_port", &peek_at_datum_on_port
      , call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("idx") = 0
      , "Peek at a datum on a port.")
    .def("grab_from_port", &grab_from_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "Grab a datum packet from a port.")
    .def("grab_value_from_port", &grab_value_from_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "Grab a value from a port.")
    .def("grab_datum_from_port", &grab_datum_from_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port")
      , "Grab a datum from a port.")
    .def("push_to_port", static_cast<void (sprokit::process::*)(sprokit::process::port_t const&, sprokit::edge_datum_t const&) const>(&wrap_process::push_to_port), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("datum")
      , "Push a datum packet to a port.")
    .def("push_value_to_port", &push_value_to_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("value")
      , "Push a value to a port.")
    .def("push_datum_to_port", &push_datum_to_port, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("port"), arg("datum")
      , "Push a datum to a port.")
    .def("get_config", static_cast<kwiver::vital::config_block_sptr (sprokit::process::*)() const>(&wrap_process::get_config), call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Gets the configuration for a process.")
    .def("config_value", &config_value, call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("key")
      , "Gets a value from the configuration for a process.")
    .def("set_data_checking_level", static_cast<void (sprokit::process::*)(sprokit::process::data_check_t)>(&wrap_process::set_data_checking_level), call_guard<kwiver::vital::python::gil_scoped_release>()
      , arg("check")
      , "Set the level to which the inputs are automatically checked.")
  ;
}

void
process_trampoline
::_configure()
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _configure,
  );
}

void
process_trampoline
::_init()
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _init,
  );
}

void
process_trampoline
::_reset()
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _reset,
  );
}

void
process_trampoline
::_flush()
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _flush,
  );
}

void
process_trampoline
::_step()
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _step,
  );
}

void
process_trampoline
::_reconfigure(kwiver::vital::config_block_sptr const& config)
{
  VITAL_PYBIND11_OVERLOAD(
    void,
    process,
    _reconfigure,
    config
  );
}

sprokit::process::properties_t
process_trampoline
::_properties_over() const
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::properties_t,
    process,
    _properties,
  );
}

sprokit::process::properties_t
process_trampoline
::_properties() const
{
  sprokit::process::properties_t consts = _properties_over();
  consts.insert("_python");
  return consts;
}

sprokit::process::ports_t
process_trampoline
::_input_ports() const
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::ports_t,
    process,
    _input_ports,
  );
}

sprokit::process::ports_t
process_trampoline
::_output_ports() const
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::ports_t,
    process,
    _output_ports,
  );
}

sprokit::process::port_info_t
process_trampoline
::_input_port_info(port_t const& port)
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::port_info_t,
    process,
    _input_port_info,
    port
  );
}

sprokit::process::port_info_t
process_trampoline
::_output_port_info(port_t const& port)
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::port_info_t,
    process,
    _output_port_info,
    port
  );
}

bool
process_trampoline
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  VITAL_PYBIND11_OVERLOAD(
    bool,
    process,
    _set_input_port_type,
    port, new_type
  );
}

bool
process_trampoline
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  VITAL_PYBIND11_OVERLOAD(
    bool,
    process,
    _set_output_port_type,
    port, new_type
  );
}

kwiver::vital::config_block_keys_t
process_trampoline
::_available_config() const
{
  VITAL_PYBIND11_OVERLOAD(
    kwiver::vital::config_block_keys_t,
    process,
    _available_config,
  );
}

sprokit::process::conf_info_t
process_trampoline
::_config_info(kwiver::vital::config_block_key_t const& key)
{
  VITAL_PYBIND11_OVERLOAD(
    sprokit::process::conf_info_t,
    process,
    _config_info,
    key
  );
}

void
declare_input_port_2(sprokit::process &self, sprokit::process::port_t const& port, sprokit::process::port_info_t const& port_info)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_input_port(port, port_info);
}

void
declare_input_port_5(sprokit::process &self,
                     sprokit::process::port_t const& port,
                     sprokit::process::port_type_t const& type_,
                     sprokit::process::port_flags_t const& flags_,
                     sprokit::process::port_description_t const& description_,
                     sprokit::process::port_frequency_t const& frequency_)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_input_port(port, type_, flags_, description_, frequency_);
}

void
declare_output_port_2(sprokit::process &self, sprokit::process::port_t const& port, sprokit::process::port_info_t const& port_info)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_output_port(port, port_info);
}

void
declare_output_port_5(sprokit::process &self,
                      sprokit::process::port_t const& port,
                      sprokit::process::port_type_t const& type_,
                      sprokit::process::port_flags_t const& flags_,
                      sprokit::process::port_description_t const& description_,
                      sprokit::process::port_frequency_t const& frequency_)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_output_port(port, type_, flags_, description_, frequency_);
}

void
declare_configuration_key_2(sprokit::process &self,
                            kwiver::vital::config_block_key_t const& key,
                            sprokit::process::conf_info_t const& info)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_configuration_key(key, info);
}

void
declare_configuration_key_3(sprokit::process &self,
                            kwiver::vital::config_block_key_t const& key,
                            kwiver::vital::config_block_value_t const& def_,
                            kwiver::vital::config_block_description_t const& description_)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_configuration_key(key, def_, description_);
}

void
declare_configuration_key_4(sprokit::process &self,
                            kwiver::vital::config_block_key_t const& key,
                            kwiver::vital::config_block_value_t const& def_,
                            kwiver::vital::config_block_description_t const& description_,
                            bool tunable_)
{
  sprokit::process* self_ptr = &self;
  ((wrap_process*) self_ptr)->declare_configuration_key(key, def_, description_, tunable_);
}

wrap_edge_datum
peek_at_port(sprokit::process &self, sprokit::process::port_t const& port, std::size_t idx)
{
  sprokit::process* self_ptr = &self;
  return wrap_edge_datum(((wrap_process*) self_ptr)->peek_at_port(port, idx));
}

wrap_edge_datum
grab_from_port(sprokit::process &self, sprokit::process::port_t const& port)
{
  sprokit::process* self_ptr = &self;
  return wrap_edge_datum(((wrap_process*) self_ptr)->grab_from_port(port));
}

object
grab_value_from_port(sprokit::process &self, sprokit::process::port_t const& port)
{
  sprokit::process* self_ptr = &self;
  sprokit::datum_t const dat = ((wrap_process*) self_ptr)->grab_datum_from_port(port);
  kwiver::vital::any const any = dat->get_datum<kwiver::vital::any>();

  //We have to explicitly list the different types we try
  object return_val = none();
  try
  {
    return_val = kwiver::vital::any_cast<object>(any);
  } catch(...){}

#define CONVERT_ANY(T) \
  try \
  { \
    return_val = cast<T>(kwiver::vital::any_cast<T>(any)); \
  } catch(...){}

  CONVERT_ANY(int)
  CONVERT_ANY(long)
  CONVERT_ANY(double)
  CONVERT_ANY(float)
  CONVERT_ANY(bool)
  CONVERT_ANY(std::string)
  CONVERT_ANY(char*)

#undef CONVERT_ANY

  return return_val;
}

sprokit::datum
grab_datum_from_port(sprokit::process &self, sprokit::process::port_t const& port)
{
  sprokit::process* self_ptr = &self;
  auto const edat = ((wrap_process*) self_ptr)->grab_from_port(port);
  sprokit::datum dat = *edat.datum;
  return dat;
}

sprokit::datum
peek_at_datum_on_port(sprokit::process &self, sprokit::process::port_t const& port,
                      std::size_t idx)
{
  sprokit::process* self_ptr = &self;
  sprokit::datum dat = *((wrap_process*) self_ptr)->peek_at_datum_on_port(port, idx);
  return dat;
}

void
push_datum_to_port(sprokit::process &self, sprokit::process::port_t const& port, sprokit::datum const& dat)
{
  sprokit::process* self_ptr = &self;
  return ((wrap_process*) self_ptr)->push_datum_to_port(port, std::make_shared<sprokit::datum>(dat));
}

void
push_value_to_port(sprokit::process &self, sprokit::process::port_t const& port, object const& obj)
{
  kwiver::vital::any const any = obj;
  sprokit::datum_t const dat = sprokit::datum::new_datum(any);

  sprokit::process* self_ptr = &self;
  return ((wrap_process*) self_ptr)->push_datum_to_port(port, dat);

}

std::string
config_value(sprokit::process &self, kwiver::vital::config_block_key_t const& key)
{
  sprokit::process* self_ptr = &self;
  return ((wrap_process*) self_ptr)->config_value<std::string>(key);
}
