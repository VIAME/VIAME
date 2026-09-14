###
# Segmentation
#
# Setting a detection's mask, and the interactive service that does it from
# a few clicks. All python so far: the C++ segmenters -- grabcut, watershed,
# the SAM family -- are still in `plugins/` and join this later in phase 2,
# and the polygon-and-mask geometry they share went to `image_ops` in
# P2-T04, one level down, because `file_io`'s CSV readers need it too.
#
# There is no `viame_segmentation` target yet for that reason: a library
# with no C++ in it is a target that links nothing.
##

if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( segmentation )
endif()
