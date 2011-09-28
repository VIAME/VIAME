/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/schedule.h>

extern "C"
{
#include <lua.h>
}

#include <luabind/luabind.hpp>
#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file schedule.cxx
 *
 * \brief Lua bindings for \link vistk::schedule\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_schedule(lua_State* L);

}

using namespace luabind;

class wrap_schedule
  : public vistk::schedule
  , public wrap_base
{
  public:
    wrap_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe);
    ~wrap_schedule();

    void start();
    void wait();
    void stop();
};

int
luaopen_vistk_pipeline_schedule(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::schedule, wrap_schedule, vistk::schedule_t>("lua_schedule")
        .def(constructor<vistk::config_t, vistk::pipeline_t>())
        .def("start", &vistk::schedule::start)
        .def("wait", &vistk::schedule::wait)
        .def("stop", &vistk::schedule::stop)
    ]
  ];

  return 0;
}

wrap_schedule
::wrap_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe)
  : vistk::schedule(config, pipe)
{
}

wrap_schedule
::~wrap_schedule()
{
}

void
wrap_schedule
::start()
{
  call<void>("start");
}

void
wrap_schedule
::wait()
{
  call<void>("wait");
}

void
wrap_schedule
::stop()
{
  call<void>("stop");
}
