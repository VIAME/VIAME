#
# Checks compiler configuration and features available
#

function(vital_check_feature NAME TEST)
  try_compile(_vital_check_feature_${NAME}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/configcheck/${TEST}
    CMAKE_FLAGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD})

  set(VITAL_USE_${NAME} ${_vital_check_feature_${NAME}} PARENT_SCOPE)
endfunction()

macro(vital_check_required_feature NAME TEST MESSAGE)
  vital_check_feature(${NAME} ${TEST})
  if (NOT VITAL_USE_${NAME})
    message(SEND_ERROR "Required C++ feature '${MESSAGE}' is not available")
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

###
# See if demangle API is supported
vital_check_feature(ABI_DEMANGLE demangle.cxx)
