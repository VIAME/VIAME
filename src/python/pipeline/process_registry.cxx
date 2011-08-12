/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * \file process_registry.cxx
 *
 * \brief Python bindings for \link vistk::process_registry\endlink.
 */

using namespace boost::python;

static void register_process(vistk::process_registry_t self,
                             vistk::process_registry::type_t const& type,
                             vistk::process_registry::description_t const& desc,
                             object obj);

static void translator(vistk::process_registry_exception const& e);

BOOST_PYTHON_MODULE(process_registry)
{
  register_exception_translator<
    vistk::process_registry_exception>(translator);

  class_<vistk::process_registry::type_t>("ProcessType");
  class_<vistk::process_registry::description_t>("ProcessDescription");
  class_<vistk::process_registry::types_t>("ProcessTypes")
    .def(vector_indexing_suite<vistk::process_registry::types_t>())
  ;

  class_<vistk::process, vistk::process_t, boost::noncopyable>("Process"
    , no_init)
    .def("init", &vistk::process::init)
    .def("step", &vistk::process::step)
    .def("is_reentrant", &vistk::process::is_reentrant)
    .def("connect_input_port", &vistk::process::connect_input_port)
    .def("connect_output_port", &vistk::process::connect_output_port)
    .def("input_ports", &vistk::process::input_ports)
    .def("output_ports", &vistk::process::output_ports)
    .def("input_port_info", &vistk::process::input_port_info)
    .def("output_port_info", &vistk::process::output_port_info)
    .def("available_config", &vistk::process::available_config)
    .def("config_info", &vistk::process::config_info)
    .def("name", &vistk::process::name)
    .def("type", &vistk::process::type)
    .def_readonly("port_heartbeat", &vistk::process::port_heartbeat)
    .def_readonly("config_name", &vistk::process::config_name)
    .def_readonly("config_type", &vistk::process::config_type)
    .def_readonly("type_any", &vistk::process::type_any)
    .def_readonly("type_none", &vistk::process::type_none)
    .def_readonly("flag_output_const", &vistk::process::flag_output_const)
    .def_readonly("flag_input_mutable", &vistk::process::flag_input_mutable)
    .def_readonly("flag_required", &vistk::process::flag_required)
  ;
  class_<vistk::processes_t>("Processes")
    .def(vector_indexing_suite<vistk::processes_t>())
  ;

  class_<vistk::process_registry, vistk::process_registry_t, boost::noncopyable>("ProcessRegistry"
    , no_init)
    .def("self", &vistk::process_registry::self)
    .staticmethod("self")
    .def("register_process", &register_process)
    .def("create_process", &vistk::process_registry::create_process)
    .def("types", &vistk::process_registry::types)
    .def("description", &vistk::process_registry::description)
  ;
}

class python_process_wrapper
{
  public:
    python_process_wrapper(object obj);
    ~python_process_wrapper();

    vistk::process_t operator () (vistk::config_t const& config);
  private:
    object const m_obj;
};

void
register_process(vistk::process_registry_t self,
                 vistk::process_registry::type_t const& type,
                 vistk::process_registry::description_t const& desc,
                 object obj)
{
  python_process_wrapper wrap(obj);

  self->register_process(type, desc, wrap);
}

void
translator(vistk::process_registry_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
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
  return extract<vistk::process_t>(m_obj(config));
}
