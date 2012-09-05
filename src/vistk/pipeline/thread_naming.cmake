set(thread_naming_defines)

include(CMakePushCheckState)
include(CheckFunctionExists)

if (CMAKE_USE_PTHREADS_INIT)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  check_function_exists(pthread_setname_np HAVE_PTHREAD_SETNAME_NP)
  check_function_exists(pthread_set_name_np HAVE_PTHREAD_SET_NAME_NP)
  cmake_pop_check_state()

  if (HAVE_PTHREAD_SETNAME_NP OR HAVE_PTHREAD_SET_NAME_NP)
    set(thread_naming_defines
      ${thread_naming_defines}
      HAVE_PTHREAD_NAMING)
  endif ()

  if (HAVE_PTHREAD_SETNAME_NP)
    set(thread_naming_defines
      ${thread_naming_defines}
      HAVE_PTHREAD_SETNAME_NP)

    set(cmakefiles_dir
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
    set(pthread_setname_np_takes_id_path
      "${cmakefiles_dir}/pthread_setname_np_takes_id.cxx")
    file(WRITE "${pthread_setname_np_takes_id_path}"
"
#include <pthread.h>

#include <cstdlib>

int
main()
{
  pthread_t const tid = pthread_self();

  pthread_setname_np(tid, \"test\");

  return EXIT_SUCCESS;
}
")
    try_compile(PTHREAD_SETNAME_NP_TAKES_ID
      "${cmakefiles_dir}/pthread_setname_np_takes_id"
      "${pthread_setname_np_takes_id_path}"
      CMAKE_FLAGS
        "-DLINK_LIBRARIES=${CMAKE_THREAD_LIBS_INIT}")

    if (PTHREAD_SETNAME_NP_TAKES_ID)
      set(thread_naming_defines
        ${thread_naming_defines}
        PTHREAD_SETNAME_NP_TAKES_ID)
    endif ()
  endif ()

  if (HAVE_PTHREAD_SET_NAME_NP)
    set(thread_naming_defines
      ${thread_naming_defines}
      HAVE_PTHREAD_SET_NAME_NP)
  endif ()
endif ()

cmake_push_check_state()
check_function_exists(setproctitle HAVE_SETPROCTITLE)
cmake_pop_check_state()

if (HAVE_SETPROCTITLE)
  set(thread_naming_defines
    ${thread_naming_defines}
    HAVE_SETPROCTITLE)
endif ()
