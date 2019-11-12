#
# Checks compiler configuration and features available
#

function(vital_check_feature NAME TEST)
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

function(vital_check_feature_run    NAME TEST)
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

macro(vital_check_required_feature NAME TEST MESSAGE)
  message(STATUS "checking ${NAME} ${TEST}")
  vital_check_feature(${NAME} ${TEST})
  if (NOT VITAL_USE_${NAME})
    message(SEND_ERROR "Required C++ feature '${MESSAGE}' is not available")
  endif()
endmacro()

macro(vital_check_optional_feature NAME TEST MESSAGE)
  message(STATUS "checking ${NAME} ${TEST}")
  vital_check_feature_run(${NAME} ${TEST})
  if (NOT VITAL_USE_${NAME})
    message(STATUS "Required C++ feature '${MESSAGE}' is not available")
  endif()
endmacro()

# C++11 is required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

vital_check_required_feature(CPP_AUTO         auto.cxx            "auto type specifier")
vital_check_required_feature(CPP_CONSTEXPR    constexpr.cxx       "constant expressions")
vital_check_required_feature(CPP_DEFAULT_CTOR default-ctor.cxx    "explicitly defaulted constructors")
vital_check_required_feature(CPP_FINAL        final.cxx           "final keyword")
vital_check_required_feature(CPP_NOEXCEPT     throw-noexcept.cxx  "noexcept specifier")
vital_check_required_feature(CPP_RANGE_FOR    range-for.cxx       "range-based for")
vital_check_required_feature(STD_CHRONO       std_chrono.cxx      "std::chrono")
vital_check_required_feature(STD_NULLPTR      null_ptr.cxx        "nullptr")
vital_check_optional_feature(STD_REGEX        std_regex.cxx       "std::regex")

###
# See if demangle API is supported
vital_check_feature(ABI_DEMANGLE demangle.cxx)
