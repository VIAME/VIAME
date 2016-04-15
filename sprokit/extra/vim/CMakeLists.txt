project(sprokit_extra_vim)

set(VIM_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/${PREFIX}/share/vim/vimfiles)

install(
  FILES       pipe.vim
  DESTINATION "${VIM_INSTALL_PATH}/syntax"
  COMPONENT   metadata)
