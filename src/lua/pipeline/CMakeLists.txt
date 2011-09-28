project(vistk_lua_pipeline)

add_vistk_lua_library(config
  vistk/pipeline
  config.cxx)
target_link_libraries(lua-config
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(datum
  vistk/pipeline
  datum.cxx)
target_link_libraries(lua-datum
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(edge
  vistk/pipeline
  edge.cxx)
target_link_libraries(lua-edge
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(modules
  vistk/pipeline
  modules.cxx)
target_link_libraries(lua-modules
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(pipeline
  vistk/pipeline
  pipeline.cxx)
target_link_libraries(lua-pipeline
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(process
  vistk/pipeline
  process.cxx)
target_link_libraries(lua-process
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(schedule
  vistk/pipeline
  schedule.cxx)
target_link_libraries(lua-schedule
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(stamp
  vistk/pipeline
  stamp.cxx)
target_link_libraries(lua-stamp
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(utils
  vistk/pipeline
  utils.cxx)
target_link_libraries(lua-utils
  vistk_pipeline
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})
