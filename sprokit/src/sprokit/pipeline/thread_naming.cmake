set(thread_naming_defines)

include(CMakePushCheckState)
include(CheckFunctionExists)

if (CMAKE_USE_PTHREADS_INIT)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  check_function_exists(pthread_setname_np have_pthread_setname_np)
  check_function_exists(pthread_set_name_np have_pthread_set_name_np)
  cmake_pop_check_state()

  if (have_pthread_setname_np OR have_pthread_set_name_np)
    set(thread_naming_defines
      ${thread_naming_defines}
      HAVE_PTHREAD_NAMING)
  endif ()

  if (have_pthread_setname_np)
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
    try_compile(pthread_setname_np_takes_id
      "${cmakefiles_dir}/pthread_setname_np_takes_id"
      "${pthread_setname_np_takes_id_path}"
      CMAKE_FLAGS
        "-DLINK_LIBRARIES=${CMAKE_THREAD_LIBS_INIT}")

    if (pthread_setname_np_takes_id)
      set(thread_naming_defines
        ${thread_naming_defines}
        PTHREAD_SETNAME_NP_TAKES_ID)
    endif ()
  endif ()

  if (have_pthread_set_name_np)
    set(thread_naming_defines
      ${thread_naming_defines}
      HAVE_PTHREAD_SET_NAME_NP)
  endif ()
endif ()

cmake_push_check_state()
check_function_exists(setproctitle have_setproctitle)
cmake_pop_check_state()

if (have_setproctitle)
  set(thread_naming_defines
    ${thread_naming_defines}
    HAVE_SETPROCTITLE)
endif ()
