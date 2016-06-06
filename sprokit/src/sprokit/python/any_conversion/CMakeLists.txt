project(sprokit_python_any_conversion)

set(python_any_conversion_srcs
  registration.cxx)

set(python_any_conversion_headers
  any_conversion-config.h
  prototypes.h
  registration.h)

# XXX: Force building shared (see src/sprokit/pipeline/CMakeLists.txt as to why
# this must be done).
sprokit_add_library(sprokit_python_any_conversion SHARED
  ${python_any_conversion_srcs}
  ${python_any_conversion_headers})
target_link_libraries(sprokit_python_any_conversion
  PUBLIC        sprokit_python_util
                ${${boost_python_library_var}}
                ${PYTHON_LIBRARIES}
  PRIVATE       ${Boost_THREAD_LIBRARY}
                ${Boost_CHRONO_LIBRARY}
                ${Boost_SYSTEM_LIBRARY}
                ${Boost_SYSTEM_LIBRARY}
                ${CMAKE_THREAD_LIBS_INIT}
                vital_logger)
set_target_properties(sprokit_python_any_conversion
  PROPERTIES
    VERSION       ${KWIVER_VERSION}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_PYTHON_ANY_CONVERSION_LIB)

sprokit_install_headers(sprokit/python/any_conversion ${python_any_conversion_headers})

sprokit_configure_pkgconfig(sprokit-python_any_conversion     PYTHON_VERSION)
