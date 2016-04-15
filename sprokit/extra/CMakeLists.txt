project(sprokit_extra)

option(SPROKIT_ENABLE_EXTRAS_VIM "Enable extra files for Vim" OFF)
if (SPROKIT_ENABLE_EXTRAS_VIM)
  add_subdirectory(vim)
endif ()
