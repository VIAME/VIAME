/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * \file process.cxx
 *
 * \brief Python bindings for \link vistk::process\endlink.
 */

using namespace boost::python;

static void translator(vistk::process_exception const& e);

class wrap_process
  : public vistk::process
  , public wrapper<vistk::process>
{
  public:
    wrap_process(vistk::config_t const& config);
    ~wrap_process();

    void _init();

    void _step();

    void _connect_input_port(port_t const& port, vistk::edge_ref_t edge);
    void _connect_output_port(port_t const& port, vistk::edge_ref_t edge);

    ports_t _input_ports() const;
    ports_t _output_ports() const;

    port_info_t _input_port_info(port_t const& port) const;
    port_info_t _output_port_info(port_t const& port) const;

    vistk::config::keys_t _available_config() const;

    conf_info_t _config_info(vistk::config::key_t const& key) const;

    void _mark_as_complete();
    vistk::stamp_t _heartbeat_stamp() const;

    bool _same_colored_data(vistk::edge_data_t const& data);
    bool _syncd_data(vistk::edge_data_t const& data);
    vistk::datum::datum_type_t _max_status(vistk::edge_data_t const& data);
    void _push_to_edges(vistk::edge_group_t const& edges, vistk::edge_datum_t const& dat);
    vistk::edge_datum_t _grab_from_edge_ref(vistk::edge_ref_t const& edge);
};

BOOST_PYTHON_MODULE(process)
{
  register_exception_translator<
    vistk::process_exception>(translator);

  class_<vistk::process::name_t>("ProcessName");
  class_<vistk::process::names_t>("ProcessNames")
    .def(vector_indexing_suite<vistk::process::names_t>())
  ;
  class_<vistk::process::port_description_t>("PortDescription");
  class_<vistk::process::port_t>("Port");
  class_<vistk::process::ports_t>("Ports")
    .def(vector_indexing_suite<vistk::process::ports_t>())
  ;
  class_<vistk::process::port_type_t>("PortType");
  class_<vistk::process::port_flag_t>("PortFlag");
  class_<vistk::process::port_flags_t>("PortFlags")
    .def("__len__", &vistk::process::port_flags_t::size)
    //.def("__contains__", &port_flags_contains)
    //.def("isdisjoint", &port_flags_isdisjoint)
    //.def("issubset", &port_flags_issubset)
    //.def("issuperset", &port_flags_issuperset)
    //.def("union", &port_flags_union)
    //.def("intersection", &port_flags_intersection)
    //.def("difference", &port_flags_difference)
    //.def("symmetric_difference", &port_flags_symmetric_difference)
    //.def("copy", &port_flags_copy)
    //.def("update", &port_flags_update)
    //.def("intersection_update", &port_flags_intersection_update)
    //.def("difference_update", &port_flags_difference_update)
    //.def("symmetric_difference_update", &port_flags_symmetric_difference_update)
    //.def("add", &port_flags_add)
    //.def("remove", &port_flags_remove)
    //.def("discard", &port_flags_discard)
    //.def("pop", &port_flags_discard)
    .def("clear", &vistk::process::port_flags_t::clear)
  ;
  class_<vistk::process::port_addr_t>("PortAddr");
  class_<vistk::process::port_addrs_t>("PortAddrs")
    .def(vector_indexing_suite<vistk::process::port_addrs_t>())
  ;

  /*
   *class_<vistk::process::port_info, vistk::process::port_info_t>("PortInfo", no_init)
   *  .def(init<vistk::process::port_type_t, vistk::process::port_flags_t, vistk::process::port_description_t>())
   *  .def_readonly("type", &vistk::process::port_info::type)
   *  .def_readonly("flags", &vistk::process::port_info::flags)
   *  .def_readonly("description", &vistk::process::port_info::description)
   *;
   */
  /*
   *class_<vistk::process::conf_info, vistk::process::conf_info_t>("ConfInfo", no_init)
   *  .def(init<vistk::config::value_t, vistk::config::description_t>())
   *  .def_readonly("default", &vistk::process::conf_info::def)
   *  .def_readonly("description", &vistk::process::conf_info::description)
   *;
   */

  class_<wrap_process, boost::noncopyable>("Process"
    , no_init)
    .def(init<vistk::config_t>())
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
    .def("_init", &wrap_process::_init)
    .def("_step", &wrap_process::_step)
    .def("_connect_input_port", &wrap_process::_connect_input_port)
    .def("_connect_output_port", &wrap_process::_connect_output_port)
    .def("_input_ports", &wrap_process::_input_ports)
    .def("_output_ports", &wrap_process::_output_ports)
    .def("_input_port_info", &wrap_process::_input_port_info)
    .def("_output_port_info", &wrap_process::_output_port_info)
    .def("_available_config", &wrap_process::_available_config)
    .def("_config_info", &wrap_process::_config_info)
    .def("mark_as_complete", &wrap_process::_mark_as_complete)
    .def("heartbeat_stamp", &wrap_process::_heartbeat_stamp)
    .def("same_colored_data", &wrap_process::_same_colored_data)
    .def("syncd_data", &wrap_process::_syncd_data)
    .def("max_status", &wrap_process::_max_status)
    .def("push_to_edges", &wrap_process::_push_to_edges)
    .def("grab_from_edge", &wrap_process::_grab_from_edge_ref)
  ;
}

void
translator(vistk::process_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
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
::_init()
{
  override f = get_override("_init");

  if (f)
  {
    f();
  }

  process::_init();
}

void
wrap_process
::_step()
{
  override f = get_override("_step");

  if (f)
  {
    f();
  }

  process::_step();
}

void
wrap_process
::_connect_input_port(port_t const& port, vistk::edge_ref_t edge)
{
  override f = get_override("_connect_input_port");

  bool ret = false;

  if (f)
  {
    ret = f(port, edge);
  }

  if (!ret)
  {
    process::_connect_input_port(port, edge);
  }
}

void
wrap_process
::_connect_output_port(port_t const& port, vistk::edge_ref_t edge)
{
  override f = get_override("_connect_output_port");

  bool ret = false;

  if (f)
  {
    ret = f(port, edge);
  }

  if (!ret)
  {
    process::_connect_output_port(port, edge);
  }
}

vistk::process::ports_t
wrap_process
::_input_ports() const
{
  override f = get_override("_input_ports");

  if (f)
  {
    return f();
  }
  else
  {
    return process::_input_ports();
  }
}

vistk::process::ports_t
wrap_process
::_output_ports() const
{
  override f = get_override("_output_ports");

  if (f)
  {
    return f();
  }
  else
  {
    return process::_output_ports();
  }
}

vistk::process::port_info_t
wrap_process
::_input_port_info(port_t const& port) const
{
  override f = get_override("_input_port_info");

  if (f)
  {
    return f(port);
  }
  else
  {
    return process::_input_port_info(port);
  }
}

vistk::process::port_info_t
wrap_process
::_output_port_info(port_t const& port) const
{
  override f = get_override("_output_port_info");

  if (f)
  {
    return f(port);
  }
  else
  {
    return process::_output_port_info(port);
  }
}

vistk::config::keys_t
wrap_process
::_available_config() const
{
  override f = get_override("_available_config");

  if (f)
  {
    return f();
  }
  else
  {
    return process::_available_config();
  }
}

vistk::process::conf_info_t
wrap_process
::_config_info(vistk::config::key_t const& key) const
{
  override f = get_override("_config_info");

  if (f)
  {
    return f(key);
  }
  else
  {
    return process::_config_info(key);
  }
}

void
wrap_process
::_mark_as_complete()
{
  mark_as_complete();
}

vistk::stamp_t
wrap_process
::_heartbeat_stamp() const
{
  return heartbeat_stamp();
}

bool
wrap_process
::_same_colored_data(vistk::edge_data_t const& data)
{
  return same_colored_data(data);
}

bool
wrap_process
::_syncd_data(vistk::edge_data_t const& data)
{
  return syncd_data(data);
}

vistk::datum::datum_type_t
wrap_process
::_max_status(vistk::edge_data_t const& data)
{
  return max_status(data);
}

void
wrap_process
::_push_to_edges(vistk::edge_group_t const& edges, vistk::edge_datum_t const& dat)
{
  push_to_edges(edges, dat);
}

vistk::edge_datum_t
wrap_process
::_grab_from_edge_ref(vistk::edge_ref_t const& edge)
{
  return grab_from_edge_ref(edge);
}
