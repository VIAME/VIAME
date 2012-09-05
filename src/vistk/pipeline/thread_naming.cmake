set(thread_naming_defines)

include(CMakePushCheckState)
include(CheckFunctionExists)

cmake_push_check_state()
check_function_exists(setproctitle HAVE_SETPROCTITLE)
cmake_pop_check_state()

if (HAVE_SETPROCTITLE)
  set(thread_naming_defines
    ${thread_naming_defines}
    HAVE_SETPROCTITLE)
endif ()
