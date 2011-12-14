project(vistk_extra_vim)

set(VIM_VERSION "73"
  CACHE STRING "The version of Vim to install for")

set(VIM_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/${PREFIX}/share/vim/vim${VIM_VERSION})

install(
  FILES       pipe.vim
  DESTINATION "${VIM_INSTALL_PATH}/syntax"
  COMPONENT   metadata)
