
if (VIAME_ENABLE_MATLAB)
  add_subdirectory(matlab)
endif()

if (VIAME_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

add_subdirectory(cpp)
