# Compiler feature checks
#
# Kwiver's `kwiver-configcheck.cmake` probed eleven things by compiling a
# small program for each: `auto`, `constexpr`, defaulted constructors,
# `final`, `noexcept`, range-based `for`, `std::chrono`, `nullptr`,
# `std::regex`. Every one of them is C++11, VIAME builds as C++17, and a
# compiler that failed any of them could not compile a line of this tree --
# the check would have reported the problem after the problem had already
# made the configure impossible.
#
# One probe survives, and it is the only one whose answer anything reads:
# `VITAL_USE_ABI_DEMANGLE`, which `util/demangle.cxx` uses to decide whether
# `abi::__cxa_demangle` exists. That is a real difference between toolchains
# rather than a standard everyone has.

include_guard( GLOBAL )

#+
# Does this toolchain compile the given probe?
#-
function( viame_check_feature NAME TEST )
  if( DEFINED VITAL_USE_${NAME} )
    return()
  endif()
  try_compile( VITAL_USE_${NAME}
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/configcheck/${TEST}"
    CMAKE_FLAGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    )
endfunction()

viame_check_feature( ABI_DEMANGLE demangle.cxx )
