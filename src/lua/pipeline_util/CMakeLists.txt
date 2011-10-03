project(vistk_lua_pipeline_util)


add_vistk_lua_library(export
  vistk/pipeline_util
  export.cxx)
target_link_libraries(lua-export
  vistk_pipeline
  vistk_pipeline_util
  luastream
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

add_vistk_lua_library(load
  vistk/pipeline_util
  load.cxx)
target_link_libraries(lua-load
  vistk_pipeline
  vistk_pipeline_util
  luastream
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

make_lua_init_file(vistk/pipeline_util)
#  export
#  load)
