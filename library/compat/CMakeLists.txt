###
# The namespace names VIAME had before phase 11
#
# Header only, and nothing in VIAME includes it: it is installed for code
# outside VIAME that still writes `kwiver::vital` or `sprokit`. Open decision
# 8 keeps it for one release.
##

viame_install_headers( kwiver.h SUBDIR viame/compat )
