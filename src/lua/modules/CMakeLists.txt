set(modules_lua_srcs
  registration.cxx)

add_vistk_plugin(modules_lua
  MAKE_VISTK_MODULES_LUA_LIB
  ${modules_lua_srcs})
target_link_libraries(modules_lua
  vistk_pipeline
  ${LUA_LIBRARIES})
