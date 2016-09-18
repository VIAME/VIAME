
if( APPLE )
  set( SETUP_READLINK_CMD "this_dir=$(cd \"$(dirname \"$BASH_SOURCE[0]\")\" && pwd)" )
  set( SETUP_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$this_dir/lib:$DYLD_FALLBACK_LIBRARY_PATH" )
else()
  set( SETUP_READLINK_CMD "this_dir=$(readlink -f $(dirname $BASH_SOURCE[0]))" )
  set( SETUP_LIBRARY_PATH "export LD_LIBRARY_PATH=$this_dir/lib:$LD_LIBRARY_PATH" )
endif()

if( VIAME_ENABLE_MATLAB )
  get_filename_component( Matlab_LIBRARY_DIR "${Matlab_ENG_LIBRARY}" DIRECTORY )

  if( APPLE )
    set( SETUP_MATLAB_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:${Matlab_LIBRARY_DIR}" )
  else()
    set( SETUP_MATLAB_LIBRARY_PATH "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/lib:${Matlab_LIBRARY_DIR}" )
  endif()
endif()
