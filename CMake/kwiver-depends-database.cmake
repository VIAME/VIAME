# Optional find and confgure database dependency

option( KWIVER_ENABLE_DATABASE
  "Enable VXL dependent code and plugins"
  OFF
  )

if( KWIVER_ENABLE_DATABASE )
  find_package(CppDB REQUIRED)
  add_definitions(-DMODULE_PATH="${CppDB_LIB_DIR}")
  add_definitions(-DHAS_CPPDB)
  include_directories( SYSTEM ${CppDB_INCLUDE_DIR} )
endif()
