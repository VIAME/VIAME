#
# Compiler flags specific to use with GCC
#

viame_check_compiler_flag( -fvisibility=hidden )
viame_check_compiler_flag( -Wall )
viame_check_compiler_flag( -Werror=return-type )
# viame_check_compiler_flag( -Werror=non-virtual-dtor )
viame_check_compiler_flag( -Werror=narrowing )
viame_check_compiler_flag( -Werror=init-self )
viame_check_compiler_flag( -Werror=reorder )
# viame_check_compiler_flag( -Werror=overloaded-virtual )
# viame_check_compiler_flag( -Werror=cast-qual )

# Linker flags, on the link line rather than in the compile flags. They were
# in the compile flags, where the compiler ignores them; they reached the
# linker only because CMake also puts CMAKE_CXX_FLAGS on the link command.
# Every target in this tree links with the C++ driver, so saying it here
# applies them exactly where they already applied.
viame_check_linker_flag( -Wl,--no-undefined )
viame_check_linker_flag( -Wl,--copy-dt-needed-entries )
