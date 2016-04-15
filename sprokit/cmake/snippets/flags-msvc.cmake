option(SPROKIT_ENABLE_DLLHELL_WARNINGS "Enables warnings about DLL visibility" OFF)
if (NOT SPROKIT_ENABLE_DLLHELL_WARNINGS)
  # C4251: STL interface warnings
  sprokit_want_compiler_flag(/wd4251)
  # C4275: Inheritance warnings
  sprokit_want_compiler_flag(/wd4275)
endif ()

option(SPROKIT_ENABLE_ANALYZE "Enables MSVC static analysis" OFF)
if (SPROKIT_ENABLE_ANALYZE)
  sprokit_want_compiler_flag(/analyze)
endif ()

sprokit_want_compiler_flag(/W3)

# -----------------------------------------------------------------------------
# Disable deprecation warnings for standard C and STL functions in VS2005 and
# later
# -----------------------------------------------------------------------------
# XXX(msvc): 1400
if (MSVC_VERSION GREATER 1400 OR
    MSVC_VERSION EQUAL 1400)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
endif ()

# Prevent namespace pollution.
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-DNOMINMAX)
add_definitions(-DWINDOWS_EXTRA_LEAN)
