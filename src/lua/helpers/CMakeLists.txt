project(vistk_lua_helpers)

add_library(lua_convert_any STATIC
  lua_convert_any.cxx)
target_link_libraries(lua_convert_any
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

if (NOT WIN32)
  set_target_properties(lua_convert_any
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)

add_library(luastream STATIC
  luastream.cxx)
target_link_libraries(luastream
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})

if (NOT WIN32)
  set_target_properties(luastream
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
