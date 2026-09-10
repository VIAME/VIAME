#
# Checks compiler configuration and features available
#

function(kwiver_check_feature NAME TEST)
  if(DEFINED VITAL_USE_${NAME})
    return()
  endif()
  try_compile(VITAL_USE_${NAME}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/configcheck/${TEST}
    CMAKE_FLAGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD})
endfunction()

function(kwiver_check_feature_run    NAME TEST)
  if(DEFINED VITAL_USE_${NAME})
    return()
  endif()
  try_run( run_res comp_res
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/configcheck/${TEST}
    OUTPUT_VARIABLE comp_output
    CMAKE_FLAGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD})

  if ( comp_res AND (run_res EQUAL 0) )
    set(VITAL_USE_${NAME} TRUE PARENT_SCOPE )
  else()
    set(VITAL_USE_${NAME} FALSE PARENT_SCOPE)
  endif()

endfunction()

macro(kwiver_check_required_feature NAME TEST MESSAGE)
  message(STATUS "checking ${NAME} ${TEST}")
  kwiver_check_feature(${NAME} ${TEST})
  if (NOT VITAL_USE_${NAME})
    message(SEND_ERROR "Required C++ feature '${MESSAGE}' is not available")
  endif()
endmacro()

macro(kwiver_check_optional_feature NAME TEST MESSAGE)
  message(STATUS "checking ${NAME} ${TEST}")
  kwiver_check_feature_run(${NAME} ${TEST})
  if (NOT VITAL_USE_${NAME})
    message(STATUS "Required C++ feature '${MESSAGE}' is not available")
  endif()
endmacro()

# failing on linux unpredicatably, disable.
# kwiver_check_required_feature(CPP_AUTO         auto.cxx            "auto type specifier")
kwiver_check_required_feature(CPP_CONSTEXPR    constexpr.cxx       "constant expressions")
kwiver_check_required_feature(CPP_DEFAULT_CTOR default-ctor.cxx    "explicitly defaulted constructors")
kwiver_check_required_feature(CPP_FINAL        final.cxx           "final keyword")
kwiver_check_required_feature(CPP_NOEXCEPT     throw-noexcept.cxx  "noexcept specifier")
kwiver_check_required_feature(CPP_RANGE_FOR    range-for.cxx       "range-based for")
kwiver_check_required_feature(STD_CHRONO       std_chrono.cxx      "std::chrono")
kwiver_check_required_feature(STD_NULLPTR      null_ptr.cxx        "nullptr")
kwiver_check_optional_feature(STD_REGEX        std_regex.cxx       "std::regex")

###
# See if demangle API is supported
kwiver_check_feature(ABI_DEMANGLE demangle.cxx)
