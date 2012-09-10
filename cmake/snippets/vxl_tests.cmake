# Check for settings from VXL that are incompatible with flags.

set(cmakefiles_dir
  "${vistk_binary_dir}/CMakeFiles")

if (VISTK_ENABLE_PEDANTIC)
  if (CMAKE_CROSSCOMPILING)
    message(WARNING "The `-pedantic` flag is not recommended when "
      "cross-compiling.")
  else ()
    set(vxl_has_float_decls_path
      "${cmakefiles_dir}/vxl_has_float_decls.cxx")
    file(WRITE "${vxl_has_float_decls_path}"
"
#include <vcl_config_compiler.h>

#include <cstdlib>

int
main()
{
#if VCL_CAN_STATIC_CONST_INIT_FLOAT
#error \"VCL_CAN_STATIC_CONST_INIT_FLOAT is defined\"
#endif

  return EXIT_SUCCESS;
}
")

    try_compile(VXL_HAS_FLOAT_DECLS_COMPILE
      "${vistk_binary_dir}"
      "${vxl_has_float_decls_path}"
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${VXL_VCL_INCLUDE_DIR}")

    if (NOT VXL_HAS_FLOAT_DECLS_COMPILE)
      message(WARNING "VXL was compiled such that float declarations are in "
        "header files, but the `-pedantic` flag does not allow this, please "
        "continue at your own risk.")
    endif ()
  endif ()
endif ()
