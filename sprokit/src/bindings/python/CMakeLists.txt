#
#
#

project(sprokit_python)

set(boost_python_library     python)

if (KWIVER_ENABLE_PYTHON3)
  set(boost_python_library    python3)
endif ()

string(TOUPPER ${boost_python_library} boost_python_library_upper)
set(boost_python_library_var
  Boost_${boost_python_library_upper}_LIBRARY)

find_package(Boost ${sprokit_boost_version} REQUIRED
  COMPONENTS
    iostreams
    ${boost_python_library})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

sprokit_use_python_fixes()

###
#
function (_sprokit_add_python_library name modpath)
  _sprokit_create_safe_modpath(${modpath} safe_modpath)

  sprokit_add_python_library(${name} ${modpath}
    ${ARGN})

  sprokit_require_python_fixes(python-${safe_modpath}-${name})
endfunction ()

###
#
set(copyright_header
"#ckwg +28
# Copyright 2015 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)

add_subdirectory(sprokit)
add_subdirectory(modules)
add_subdirectory(processes)
add_subdirectory(schedulers)
add_subdirectory(test)

set(python_noarch FALSE)
sprokit_create_python_init(sprokit)
set(python_noarch TRUE)
sprokit_create_python_init(sprokit)
