# Optionally enable OpenMP

# OpenMP is somewhat broken on macOS and not easily supported until
# CMake 3.12+ is a minimum requirement of KWIVER
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if( APPLE )
  set( OPENMP_DEFAULT OFF )
elseif()
  set( OPENMP_DEFAULT ON )
endif()

option( KWIVER_ENABLE_OPENMP
  "Enable OpenMP for parallel processing"
  ${OPENMP_DEFAULT}
  )

if( KWIVER_ENABLE_OPENMP )
  find_package( OpenMP REQUIRED )
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
