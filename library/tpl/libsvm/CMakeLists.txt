# libsvm, vendored
#
# Two files of the distribution and its copyright notice. Everything else
# libsvm ships -- the java, matlab and python bindings, the command line
# train, predict and scale tools, svm-toy, the Windows project files and the
# example data -- is not built here and is not carried.
#
# **This copy is not stock libsvm.** It carries VIAME's `HISTOGRAM` and `NMI`
# kernel types, which the SVM refiner and the IQR session both select, and
# which have never existed upstream. That is why it is vendored rather than
# found: a system libsvm would compile and then refuse the kernel every
# shipped model was trained with.

set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

add_library( viame_libsvm STATIC svm.cpp svm.h )

target_include_directories( viame_libsvm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# libsvm prints its progress to stdout from inside the solver
target_compile_definitions( viame_libsvm PRIVATE _CRT_SECURE_NO_WARNINGS )

set_target_properties( viame_libsvm PROPERTIES
  OUTPUT_NAME svm
  )

add_library( viame::libsvm ALIAS viame_libsvm )

# Not installed: its code is linked into libviame, which is. It was installed
# only to join the export set, and nothing is exported any more.
