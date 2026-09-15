
if( APPLE )
  set( SETUP_READLINK_CMD "this_dir=$(cd \"$(dirname \"$BASH_SOURCE[0]\")\" && pwd)" )
  set( SETUP_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$DYLD_FALLBACK_LIBRARY_PATH" )
else()
  set( SETUP_READLINK_CMD "this_dir=$(readlink -f $(dirname $BASH_SOURCE[0]))" )
  set( SETUP_LIBRARY_PATH "export LD_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$LD_LIBRARY_PATH" )
endif()


set( SETUP_PYTHON_PATH "" )
set( SETUP_PYTHON_BAT_PATH "" )

if( VIAME_ENABLE_PYTHON )
  if( VIAME_PYTHON_STANDALONE )
    # The interpreter is part of the install, so it is wherever the install
    # is now -- not where it was configured. Ahead of a system python on PATH,
    # since the script applets run `python` by name; `lib` for libviame and
    # the extension modules, which link libpython.
    set( SETUP_PYTHON_LOCATION "export PYTHON_INSTALL_DIR=$this_dir/python" )
    set( SETUP_PYTHON_PATH
      "export PATH=$this_dir/python/bin:$PATH\nexport LD_LIBRARY_PATH=$this_dir/python/lib:$LD_LIBRARY_PATH" )
    set( SETUP_PYTHON_BAT_PATH "SET PATH=%VIAME_INSTALL%\\python;%VIAME_INSTALL%\\python\\Scripts;%PATH%" )
  else()
    get_filename_component( PYTHON_LIBRARY_DIR "${Python_LIBRARIES}" DIRECTORY )
    get_filename_component( PYTHON_ROOT_DIR "${PYTHON_LIBRARY_DIR}" DIRECTORY )

    set( SETUP_PYTHON_LOCATION "export PYTHON_INSTALL_DIR=${PYTHON_ROOT_DIR}" )
  endif()
endif()

set( SETUP_CUSTOM_TERMINAL "export PS1=\"\${PS1//\"(viame) \"/}\"\nexport PS1=\"(viame) \${PS1//\"(base) \"/}\"" )
