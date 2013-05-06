/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_exceptions.h>
#include <python/helpers/python_threading.h>

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_registry.h>

#include <sprokit/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/wrapper.hpp>

/**
 * \file process_registry.cxx
 *
 * \brief Python bindings for \link sprokit::process_registry\endlink.
 */

using namespace boost::python;

static void register_process(sprokit::process_registry_t reg,
                             sprokit::process::type_t const& type,
                             sprokit::process_registry::description_t const& desc,
                             object obj);

BOOST_PYTHON_MODULE(process_registry)
{
  class_<sprokit::process_registry::description_t>("ProcessDescription"
    , "The type for a description of a process type.");
  class_<sprokit::process_registry::module_t>("ProcessModule"
    , "The type for a process module name.");

  class_<sprokit::process, sprokit::process_t, boost::noncopyable>("Process"
    , "The base class of processes."
    , no_init)
    .def("configure", &sprokit::process::configure
      , "Configures the process.")
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
    .def_readonly("flag_input_static", &sprokit::process::flag_input_static)
    .def_readonly("flag_input_mutable", &sprokit::process::flag_input_mutable)
    .def_readonly("flag_input_nodep", &sprokit::process::flag_input_nodep)
    .def_readonly("flag_required", &sprokit::process::flag_required)
  ;
  class_<sprokit::processes_t>("Processes"
    , "A collection of processes.")
    .def(vector_indexing_suite<sprokit::processes_t>())
  ;

  class_<sprokit::process_registry, sprokit::process_registry_t, boost::noncopyable>("ProcessRegistry"
    , "A registry of all known process types."
    , no_init)
    .def("self", &sprokit::process_registry::self
      , "Returns an instance of the process registry.")
    .staticmethod("self")
    .def("register_process", &register_process
      , (arg("type"), arg("name"), arg("description"), arg("ctor"))
      , "Registers a function which creates a process of the given type.")
    .def("create_process", &sprokit::process_registry::create_process
      , (arg("type"), arg("config") = sprokit::config::empty_config())
      , "Creates a new process of the given type.")
    .def("types", &sprokit::process_registry::types
      , "A list of known process types.")
    .def("description", &sprokit::process_registry::description
      , (arg("type"))
      , "The description for the given type.")
    .def("is_module_loaded", &sprokit::process_registry::is_module_loaded
      , (arg("module"))
      , "Returns True if the module has already been loaded, False otherwise.")
    .def("mark_module_as_loaded", &sprokit::process_registry::mark_module_as_loaded
      , (arg("module"))
      , "Marks a module as loaded.")
  ;
}

class python_process_wrapper
  : python_threading
{
  public:
    python_process_wrapper(object obj);
    ~python_process_wrapper();

    sprokit::process_t operator () (sprokit::config_t const& config);
  private:
    object const m_obj;
};

void
register_process(sprokit::process_registry_t reg,
                 sprokit::process::type_t const& type,
                 sprokit::process_registry::description_t const& desc,
                 object obj)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  python_process_wrapper const wrap(obj);

  reg->register_process(type, desc, wrap);
}

python_process_wrapper
::python_process_wrapper(object obj)
  : m_obj(obj)
{
}

python_process_wrapper
::~python_process_wrapper()
{
}

sprokit::process_t
python_process_wrapper
::operator () (sprokit::config_t const& config)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  object proc;

  HANDLE_PYTHON_EXCEPTION(proc = m_obj(config))

  return extract<sprokit::process_t>(proc);
}
