/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_threading.h>

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/wrapper.hpp>

/**
 * \file process_registry.cxx
 *
 * \brief Python bindings for \link vistk::process_registry\endlink.
 */

using namespace boost::python;

static void register_process(vistk::process_registry_t reg,
                             vistk::process::type_t const& type,
                             vistk::process_registry::description_t const& desc,
                             object obj);

BOOST_PYTHON_MODULE(process_registry)
{
  class_<vistk::process_registry::description_t>("ProcessDescription"
    , "The type for a description of a process type.");
  class_<vistk::process_registry::module_t>("ProcessModule"
    , "The type for a process module name.");

  class_<vistk::process, vistk::process_t, boost::noncopyable>("Process"
    , "The base class of processes."
    , no_init)
    .def("configure", &vistk::process::configure
      , "Configures the process.")
    .def("init", &vistk::process::init
      , "Initializes the process.")
    .def("reset", &vistk::process::reset
      , "Resets the process.")
    .def("step", &vistk::process::step
      , "Steps the process for one iteration.")
    .def("constraints", &vistk::process::constraints
      , "Returns the constraints on the process.")
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
    .def_readonly("constraint_no_threads", &vistk::process::constraint_no_threads)
    .def_readonly("constraint_no_reentrancy", &vistk::process::constraint_no_reentrancy)
    .def_readonly("constraint_unsync_input", &vistk::process::constraint_unsync_input)
    .def_readonly("constraint_unsync_output", &vistk::process::constraint_unsync_output)
    .def_readonly("port_heartbeat", &vistk::process::port_heartbeat)
    .def_readonly("config_name", &vistk::process::config_name)
    .def_readonly("config_type", &vistk::process::config_type)
    .def_readonly("type_any", &vistk::process::type_any)
    .def_readonly("type_none", &vistk::process::type_none)
    .def_readonly("type_data_dependent", &vistk::process::type_data_dependent)
    .def_readonly("type_flow_dependent", &vistk::process::type_flow_dependent)
    .def_readonly("flag_output_const", &vistk::process::flag_output_const)
    .def_readonly("flag_input_mutable", &vistk::process::flag_input_mutable)
    .def_readonly("flag_input_nodep", &vistk::process::flag_input_nodep)
    .def_readonly("flag_required", &vistk::process::flag_required)
  ;
  class_<vistk::processes_t>("Processes"
    , "A collection of processes.")
    .def(vector_indexing_suite<vistk::processes_t>())
  ;

  class_<vistk::process_registry, vistk::process_registry_t, boost::noncopyable>("ProcessRegistry"
    , "A registry of all known process types."
    , no_init)
    .def("self", &vistk::process_registry::self
      , "Returns an instance of the process registry.")
    .staticmethod("self")
    .def("register_process", &register_process
      , (arg("type"), arg("name"), arg("description"), arg("ctor"))
      , "Registers a function which creates a process of the given type.")
    .def("create_process", &vistk::process_registry::create_process
      , (arg("type"), arg("config") = vistk::config::empty_config())
      , "Creates a new process of the given type.")
    .def("types", &vistk::process_registry::types
      , "A list of known process types.")
    .def("description", &vistk::process_registry::description
      , (arg("type"))
      , "The description for the given type.")
    .def("is_module_loaded", &vistk::process_registry::is_module_loaded
      , (arg("module"))
      , "Returns True if the module has already been loaded, False otherwise.")
    .def("mark_module_as_loaded", &vistk::process_registry::mark_module_as_loaded
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

    vistk::process_t operator () (vistk::config_t const& config);
  private:
    object const m_obj;
};

void
register_process(vistk::process_registry_t reg,
                 vistk::process::type_t const& type,
                 vistk::process_registry::description_t const& desc,
                 object obj)
{
  vistk::python::python_gil const gil;

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

vistk::process_t
python_process_wrapper
::operator () (vistk::config_t const& config)
{
  vistk::python::python_gil const gil;

  (void)gil;

  return extract<vistk::process_t>(m_obj(config));
}
