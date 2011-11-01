/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>

#include <lua/helpers/lua_include.h>
#include <lua/helpers/lua_convert_vector.h>
#include <lua/helpers/lua_static_member.h>

#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file schedule_registry.cxx
 *
 * \brief Lua bindings for \link vistk::schedule_registry\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_schedule_registry(lua_State* L);

}

using namespace luabind;

static void register_schedule(vistk::schedule_registry_t reg,
                              vistk::schedule_registry::type_t const& type,
                              vistk::schedule_registry::description_t const& desc,
                              object obj);

int
luaopen_vistk_pipeline_schedule_registry(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::schedule_registry::type_t>("schedule_type")
        .def(constructor<>())
    , class_<vistk::schedule_registry::description_t>("schedule_description")
        .def(constructor<>())
    , class_<vistk::schedule_registry::types_t>("schedule_types")
        .def(constructor<>())
    , class_<vistk::schedule_registry::module_t>("schedule_module")
        .def(constructor<>())
    , class_<vistk::schedule, vistk::schedule_t>("schedule")
        .def("start", &vistk::schedule::start)
        .def("wait", &vistk::schedule::wait)
        .def("stop", &vistk::schedule::stop)
    , class_<vistk::schedule_registry, vistk::schedule_registry_t>("schedule_registry")
        .scope
        [
          def("self", &vistk::schedule_registry::self)
        ]
        .def("register_schedule", &register_schedule)
        .def("create_schedule", &vistk::schedule_registry::create_schedule)
        .def("types", &vistk::schedule_registry::types)
        .def("description", &vistk::schedule_registry::description)
        .def("is_module_loaded", &vistk::schedule_registry::is_module_loaded)
        .def("mark_module_as_loaded", &vistk::schedule_registry::mark_module_as_loaded)
    ]
  ];

  lua_getfield(L, LUA_GLOBALSINDEX, "vistk");
  lua_getfield(L, -1, "pipeline");
  lua_getfield(L, -1, "schedule_registry");
  LUA_STATIC_MEMBER(L, string, vistk::schedule_registry::default_type, "default_type");
  lua_pop(L, 3);

  return 0;
}

class lua_schedule_wrapper
{
  public:
    lua_schedule_wrapper(object obj);
    ~lua_schedule_wrapper();

    vistk::schedule_t operator () (vistk::config_t const& config, vistk::pipeline_t const& pipeline);
  private:
    object const m_obj;
};

void
register_schedule(vistk::schedule_registry_t reg,
                  vistk::schedule_registry::type_t const& type,
                  vistk::schedule_registry::description_t const& desc,
                  object obj)
{
  lua_schedule_wrapper wrap(obj);

  reg->register_schedule(type, desc, wrap);
}

lua_schedule_wrapper
::lua_schedule_wrapper(object obj)
  : m_obj(obj)
{
}

lua_schedule_wrapper
::~lua_schedule_wrapper()
{
}

vistk::schedule_t
lua_schedule_wrapper
::operator () (vistk::config_t const& config, vistk::pipeline_t const& pipeline)
{
  return call_function<vistk::schedule_t>(m_obj, config, pipeline);
}
