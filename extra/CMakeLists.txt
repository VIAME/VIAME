project(vistk_extra)

option(VISTK_ENABLE_EXTRAS_VIM "Enable extra files for Vim" OFF)
if (VISTK_ENABLE_EXTRAS_VIM)
  add_subdirectory(vim)
endif ()
