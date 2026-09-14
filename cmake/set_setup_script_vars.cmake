
if( APPLE )
  set( SETUP_READLINK_CMD "this_dir=$(cd \"$(dirname \"$BASH_SOURCE[0]\")\" && pwd)" )
  set( SETUP_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$DYLD_FALLBACK_LIBRARY_PATH" )
else()
  set( SETUP_READLINK_CMD "this_dir=$(readlink -f $(dirname $BASH_SOURCE[0]))" )
  set( SETUP_LIBRARY_PATH "export LD_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$LD_LIBRARY_PATH" )
endif()


if( VIAME_ENABLE_PYTHON )
  get_filename_component( PYTHON_LIBRARY_DIR "${Python_LIBRARIES}" DIRECTORY )
  get_filename_component( PYTHON_ROOT_DIR "${PYTHON_LIBRARY_DIR}" DIRECTORY )

  set( SETUP_PYTHON_LOCATION "export PYTHON_INSTALL_DIR=${PYTHON_ROOT_DIR}" )
endif()

set( SETUP_CUSTOM_TERMINAL "export PS1=\"\${PS1//\"(viame) \"/}\"\nexport PS1=\"(viame) \${PS1//\"(base) \"/}\"" )
