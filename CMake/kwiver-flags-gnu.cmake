#
# Compiler flags specific to use with GCC
#


kwiver_check_compiler_flag( -std=c++11 -std=c++0x )
kwiver_check_compiler_flag( -fvisibility=hidden )
kwiver_check_compiler_flag( -Wall )
kwiver_check_compiler_flag( -Werror=return-type )
kwiver_check_compiler_flag( -Werror=non-virtual-dtor )
kwiver_check_compiler_flag( -Werror=narrowing )
kwiver_check_compiler_flag( -Werror=init-self )
kwiver_check_compiler_flag( -Werror=reorder )

# not supported by VXL
#kwiver_check_compiler_flag( -Werror=overloaded-virtual )
#kwiver_check_compiler_flag( -Werror=cast-qual )
