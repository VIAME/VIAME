set(cmakefiles_dir
  "${vistk_binary_dir}/CMakeFiles")

if (CMAKE_CROSSCOMPILING)
  set(BOOST_MANGLE_NAMESPACE "boost" CACHE STRING "The mangled boost namespace")
else ()
  set(boost_mangle_namespace_path
    "${cmakefiles_dir}/boost_mangle_namespace.cxx")
  file(WRITE "${boost_mangle_namespace_path}"
"
#include <boost/config.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <iostream>

#include <cstdlib>

int
main()
{
#ifdef boost
  std::cout << BOOST_PP_STRINGIZE(boost) << std::endl;
#else
  std::cout << \"boost\" << std::endl;
#endif

  return EXIT_SUCCESS;
}
")

  try_run(boost_mangle_namespace_run boost_mangle_namespace_compile
    "${vistk_binary_dir}"
    "${boost_mangle_namespace_path}"
    CMAKE_FLAGS         "-DINCLUDE_DIRECTORIES=${Boost_INCLUDE_DIRS}"
    RUN_OUTPUT_VARIABLE boost_detect_mangle_namespace)
  set(BOOST_MANGLE_NAMESPACE "${boost_detect_mangle_namespace}"
    CACHE INTERNAL "The detected mangled boost namespace")
endif ()
