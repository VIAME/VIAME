#
# Compiler flags specific to use with clang++
#

kwiver_check_compiler_flag( -std=c++11 -std=c++0x )
if(NOT APPLE)
  #MacOS produces warnings with this flag and doesn't seem to need it
  kwiver_check_compiler_flag( -pthread )
endif()
kwiver_check_compiler_flag( -fvisibility=hidden )
kwiver_check_compiler_flag( -Wall )
kwiver_check_compiler_flag( -Werror=return-type )
kwiver_check_compiler_flag( -Werror=non-virtual-dtor )
kwiver_check_compiler_flag( -Werror=narrowing )
kwiver_check_compiler_flag( -Werror=init-self )
kwiver_check_compiler_flag( -Werror=reorder )
kwiver_check_compiler_flag( -Werror=overloaded-virtual )
kwiver_check_compiler_flag( -Werror=cast-qual )
kwiver_check_compiler_flag( -Werror=vla )
