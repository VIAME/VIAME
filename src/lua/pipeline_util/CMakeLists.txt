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

make_lua_init_file(vistk/pipeline_util)
#  export)
