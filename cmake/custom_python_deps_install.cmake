message( "Running python deps auxiliary install" )

if( VIAME_PYTHON_VERSION VERSION_LESS "3.8" )
  set( TARGET_FILE "${VIAME_PYTHON_BASE}/site-packages/ubelt/util_cache.py" )

  set( SEARCH_CODE1
"import pickle" )
  set( REPL_CODE1
"import pickle5 as pickle" )
  set( SEARCH_CODE2
"import pickle5 as pickle5 as pickle" )
  set( REPL_CODE2
"import pickle5 as pickle" )

  if( EXISTS ${TARGET_FILE} )
    file( READ "${TARGET_FILE}" TARGET_FILE_DATA )
    string( REPLACE "${SEARCH_CODE1}" "${REPL_CODE1}" ADJ_FILE_DATA "${TARGET_FILE_DATA}" )
    string( REPLACE "${SEARCH_CODE2}" "${REPL_CODE2}" TARGET_FILE_DATA "${ADJ_FILE_DATA}" )
    file( WRITE "${TARGET_FILE}" "${TARGET_FILE_DATA}")
  endif()
endif()

message( "Done" )
