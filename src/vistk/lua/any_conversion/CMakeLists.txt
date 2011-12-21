project(vistk_lua_any_conversion)

set(lua_any_conversion_srcs
  registration.cxx)

set(lua_any_conversion_headers
  any_conversion-config.h
  prototypes.h
  registration.h)

add_library(vistk_lua_any_conversion
  ${lua_any_conversion_srcs})
target_link_libraries(vistk_lua_any_conversion
  ${Boost_THREAD_LIBRARY}
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})
target_link_libraries(vistk_lua_any_conversion LINK_INTERFACE_LIBRARIES
  ${LUABIND_LIBRARY}
  ${LUA_LIBRARIES})
set_target_properties(vistk_lua_any_conversion
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_LUA_ANY_CONVERSION_LIB)

install(
  FILES       ${lua_any_conversion_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/lua/any_conversion"
  COMPONENT   runtime)
install(
  TARGETS     vistk_lua_any_conversion
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
