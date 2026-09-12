# The scheduler
#
# One, since P8-T07. `thread_per_process` is `scheduler_factory::default_type`
# and so is what any pipeline that names no scheduler gets; every shipped
# `.pipe` names `pythread_per_process` instead, which is python's.
#
# `sync` and `thread_pool` are removed. `sync` produced identical output on
# one thread -- recorded, before it went, in
# `tests/library/pipeline_framework/test_scheduler.cxx` -- and `thread_pool`
# described itself as "not functional".

set( source
  registration.cxx
  thread_per_process_scheduler.cxx
  )

set( private_headers
  thread_per_process_scheduler.h
  ${CMAKE_CURRENT_BINARY_DIR}/schedulers_export.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

viame_private_header_group( ${private_headers} )

viame_add_plugin( schedulers
  SOURCES         ${source}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  vital_config
                  vital_vpm
                  ${CMAKE_THREAD_LIBS_INIT}
  KIND        PROCESSES
  )

