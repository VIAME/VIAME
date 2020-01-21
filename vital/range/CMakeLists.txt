###
# Build vital/range library
#
# A hard requirement for the vital_range library is that it not depend
# on vital library.
#

# ==================================================================
###
# These headers are installed
#
set( public_headers
  defs.h
  filter.h
  indirect.h
  integer_sequence.h
  iota.h
  sliding.h
  transform.h
  valid.h
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR   vital/range
  )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
