kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         ""
                         __init__)

add_subdirectory(vital)

if(KWIVER_ENABLE_SPROKIT)
  add_subdirectory(sprokit)
endif()

if(KWIVER_ENABLE_ARROWS)
  add_subdirectory(arrows)
endif()

if(SKBUILD)
kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/kwiver_tools.py
                         ""
                         kwiver_tools)
endif()
