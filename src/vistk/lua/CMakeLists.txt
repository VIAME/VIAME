find_package(Lua51 REQUIRED)
include_directories(SYSTEM ${LUA_INCLUDE_DIR})

find_package(Luabind REQUIRED)
include_directories(SYSTEM ${LUABIND_INCLUDE_DIR})

add_subdirectory(any_conversion)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    lua)
endif (VISTK_ENABLE_DOCUMENTATION)
