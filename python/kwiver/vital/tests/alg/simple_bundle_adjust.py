# ckwg +29
# Copyright 2020 by Kitware, Inc.
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function

from kwiver.vital.algo import BundleAdjust
from kwiver.vital.types import CameraMap
from kwiver.vital.tests.helpers import CommonConfigurationMixin

import numpy as np
from multimethod import multimethod
from unittest.mock import Mock

class SimpleBundleAdjust(CommonConfigurationMixin, BundleAdjust):
    """
    Implementation of BundleAdjust to test it

    """
    def __init__(self):
        BundleAdjust.__init__(self)

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory
    # Register Algorithm
    implementation_name  = "SimpleBundleAdjust"
    if algorithm_factory.has_algorithm_impl_name(
                            SimpleBundleAdjust.static_type_name(),
                            implementation_name):
        return
    algorithm_factory.add_algorithm( implementation_name,
                                "test simple bundle adjust",
                                 SimpleBundleAdjust )
    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
