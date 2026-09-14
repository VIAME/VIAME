###
# @template_dir@
#
# Installs every `.py` in this directory as the `viame.@template_dir@`
# package; its `__init__.py` declares what the package provides. A directory
# that also holds C++ adds this beside that library's targets.
##

if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( @template_dir@ )
endif()
