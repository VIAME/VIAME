# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python access to file_io's C++ readers.

Only the OpenCV FileStorage subset so far: `viame.file_io._opencv_yaml`.
There is no `__vital_algorithm_register__` here, since nothing in this
package registers an algorithm -- it is a library, not a plugin.
"""
