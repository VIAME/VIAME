/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file process_registry.cxx
 *
 * \brief Lua bindings for \link vistk::process_registry\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_process_registry(lua_State* L);

}

using namespace luabind;

static void register_process(vistk::process_registry_t reg,
                             vistk::process_registry::type_t const& type,
                             vistk::process_registry::description_t const& desc,
                             object obj);

int
luaopen_vistk_pipeline_process_registry(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::process_registry::type_t>("process_type")
        .def(constructor<>())
    , class_<vistk::process_registry::description_t>("process_description")
        .def(constructor<>())
    , class_<vistk::process_registry::types_t>("process_types")
        .def(constructor<>())
        /// \todo Add vector methods.
        //.def(vector_indexing_suite<vistk::process_registry::types_t>())
    , class_<vistk::process_registry::module_t>("process_module")
        .def(constructor<>())
    , class_<vistk::process, vistk::process_t>("process")
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
        //.scope
        //[
        //  def("port_heartbeat", &vistk::process::port_heartbeat)
        //, def("config_name", &vistk::process::config_name)
        //, def("config_type", &vistk::process::config_type)
        //, def("type_any", &vistk::process::type_any)
        //, def("type_none", &vistk::process::type_none)
        //, def("flag_output_const", &vistk::process::flag_output_const)
        //, def("flag_input_mutable", &vistk::process::flag_input_mutable)
        //, def("flag_required", &vistk::process::flag_required)
        //]
    , class_<vistk::processes_t>("processes")
        .def(constructor<>())
        /// \todo Add vector methods.
        //.def(vector_indexing_suite<vistk::processes_t>())
    , class_<vistk::process_registry, vistk::process_registry_t>("process_registry")
        .scope
        [
          def("self", &vistk::process_registry::self)
        ]
        .def("register_process", &register_process)
        .def("create_process", &vistk::process_registry::create_process)
        .def("types", &vistk::process_registry::types)
        .def("description", &vistk::process_registry::description)
        .def("is_module_loaded", &vistk::process_registry::is_module_loaded)
        .def("mark_module_as_loaded", &vistk::process_registry::mark_module_as_loaded)
    ]
  ];

  return 0;
}

class lua_process_wrapper
{
  public:
    lua_process_wrapper(object obj);
    ~lua_process_wrapper();

    vistk::process_t operator () (vistk::config_t const& config);
  private:
    object const m_obj;
};

void
register_process(vistk::process_registry_t reg,
                 vistk::process_registry::type_t const& type,
                 vistk::process_registry::description_t const& desc,
                 object obj)
{
  lua_process_wrapper wrap(obj);

  reg->register_process(type, desc, wrap);
}

lua_process_wrapper
::lua_process_wrapper(object obj)
  : m_obj(obj)
{
}

lua_process_wrapper
::~lua_process_wrapper()
{
}

vistk::process_t
lua_process_wrapper
::operator () (vistk::config_t const& config)
{
  return call_function<vistk::process_t>(m_obj, config);
}
