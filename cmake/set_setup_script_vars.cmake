# Both setup templates are sourced into the user's shell. $0 identifies the
# caller in Bash, so use each supported shell's source-file information.
set( SETUP_READLINK_CMD [=[
if [ -n "${BASH_VERSION:-}" ]; then
  viame_setup_source="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  viame_setup_source="${(%):-%x}"
else
  echo "setup_viame.sh must be sourced from Bash or Zsh" >&2
  return 1
fi
this_dir=$(CDPATH= cd -- "$(dirname -- "$viame_setup_source")" && pwd -P) || return 1
unset viame_setup_source
]=] )

if( APPLE )
  set( SETUP_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$DYLD_FALLBACK_LIBRARY_PATH" )
  set( SETUP_QT_PLUGIN_PATH "#export QT_PLUGIN_PATH=$this_dir/lib/qt4/plugins" )
else()
  set( SETUP_LIBRARY_PATH "export LD_LIBRARY_PATH=$this_dir/lib:$this_dir/lib64:$LD_LIBRARY_PATH" )
  set( SETUP_QT_PLUGIN_PATH "export QT_PLUGIN_PATH=$this_dir/lib/qt4/plugins" )
endif()

if( VIAME_ENABLE_MATLAB )
  get_filename_component( Matlab_LIBRARY_DIR "${Matlab_ENG_LIBRARY}" DIRECTORY )

  if( APPLE )
    set( SETUP_MATLAB_LIBRARY_PATH "export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:${Matlab_LIBRARY_DIR}" )
  else()
    set( SETUP_MATLAB_LIBRARY_PATH "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/lib:${Matlab_LIBRARY_DIR}" )
  endif()
endif()

if( VIAME_ENABLE_PYTHON )
  get_filename_component( PYTHON_LIBRARY_DIR "${Python_LIBRARIES}" DIRECTORY )
  get_filename_component( PYTHON_ROOT_DIR "${PYTHON_LIBRARY_DIR}" DIRECTORY )

  set( SETUP_PYTHON_LOCATION "export PYTHON_INSTALL_DIR=${PYTHON_ROOT_DIR}" )

  if( VIAME_BUILD_PYTHON_FROM_SOURCE )
    set( SETUP_PYTHON_LIBRARY_PATH "export PYTHON_LIBRARY=$this_dir/lib/libpython${Python_VERSION_MAJOR}.so" )
  else()
    set( SETUP_PYTHON_LIBRARY_PATH "export PYTHON_LIBRARY=${Python_LIBRARIES}" )
  endif()
endif()

set( SETUP_CUSTOM_TERMINAL "export PS1=\"\${PS1//\"(viame) \"/}\"\nexport PS1=\"(viame) \${PS1//\"(base) \"/}\"" )
