# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Python Process Registration Package for VIAME Examples.

Registration does not happen here. KWIVER's module loader walks this
package's directory, imports every module in it except __init__.py, and
calls a __sprokit_register__() hook on each one it imported. A hook defined
in this file is therefore never called.

To add your own process:
    1. Write it in its own module in this directory
    2. Give that module a __sprokit_register__() that calls
       process_factory.add_process(), guarding on
       process_factory.is_process_module_loaded() as the existing ones do

See hello_world_detector.py for the pattern.
"""
