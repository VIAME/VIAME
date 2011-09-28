project(vistk_lua_pipeline)

add_vistk_lua_library(datum
  vistk/pipeline
  datum.cxx)
target_link_libraries(lua-datum
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
