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
  integer_sequence.h
  sliding.h
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR   vital/range
  )
