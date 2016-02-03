#
# Checks compiler configuration and features available
#

##
# see if auto is supported
try_compile( success
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/configcheck/auto.cxx
  CMAKE_FLAGS
     -DCMAKE_CXX_FLAGS:STRING=#${CMAKE_CXX_FLAGS}
  OUTPUT_VARIABLE OUTPUT)

set( VITAL_USE_CPP_AUTO ${success} )

##
# see if range for is supported
try_compile( success
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/configcheck/range-for.cxx
  CMAKE_FLAGS
     -DCMAKE_CXX_FLAGS:STRING=#${CMAKE_CXX_FLAGS}
  OUTPUT_VARIABLE OUTPUT)

set( VITAL_USE_CPP_RANGE_FOR ${success} )

##
# see if default ctor is supported
try_compile( success
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/configcheck/default-ctor.cxx
  CMAKE_FLAGS
     -DCMAKE_CXX_FLAGS:STRING=#${CMAKE_CXX_FLAGS}
  OUTPUT_VARIABLE OUTPUT)

set( VITAL_USE_CPP_DEFAULT_CTOR ${success} )

##
# see if noexcept is supported
try_compile( success
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/configcheck/throw-noexcept.cxx
  CMAKE_FLAGS
     -DCMAKE_CXX_FLAGS:STRING=#${CMAKE_CXX_FLAGS}
  OUTPUT_VARIABLE OUTPUT)

set( VITAL_USE_CPP_NOEXCEPT ${success} )

##
# see if std::chrono is supported
try_compile( success
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/configcheck/std_chrono.cxx
  CMAKE_FLAGS
     -DCMAKE_CXX_FLAGS:STRING=#${CMAKE_CXX_FLAGS}
  OUTPUT_VARIABLE OUTPUT)

set( VITAL_USE_STD_CHRONO ${success} )

###
