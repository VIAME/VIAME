#
# Compiler flags specific to use with clang++
#

viame_check_compiler_flag( -std=c++11 -std=c++0x )
viame_check_compiler_flag( -fvisibility=hidden )
viame_check_compiler_flag( -Wall )
viame_check_compiler_flag( -Werror=return-type )
# viame_check_compiler_flag( -Werror=non-virtual-dtor )
viame_check_compiler_flag( -Werror=narrowing )
viame_check_compiler_flag( -Werror=init-self )
viame_check_compiler_flag( -Werror=reorder )
# viame_check_compiler_flag( -Werror=overloaded-virtual )
viame_check_compiler_flag( -Werror=cast-qual )
