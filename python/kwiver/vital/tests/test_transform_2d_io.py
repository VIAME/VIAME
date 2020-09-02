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
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF

from unittest import TestCase
from kwiver.vital.algo import Transform2DIO
import nose.tools
import tempfile
from kwiver.vital.modules import modules
from unittest.mock import Mock

class TestVitalTransform2DIO(TestCase):
    def setUp(self):
        modules.load_known_modules()
        self.instance = Transform2DIO.create("SimpleTransform2DIO")
        #TODO: Replace these mocks with the actual vital types
        # The successful tests in this file are not indicators of anything being
        # tested they are failing because the function signature does not match
        # with what pybind11 expects. We would modify these tests once the
        # vital types bindings are complete so that we can test for the desired
        # behavior
        self.transform_2d = Mock()

    @nose.tools.raises(RuntimeError)
    def test_load_nonexistant(self):
        self.instance.load("nonexistant_filename.txt")

    @nose.tools.raises(RuntimeError)
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load(directory_name)

    @nose.tools.raises(Exception)
    def test_save_nonexistant(self):
        self.instance.save("nonexistant_filename.txt",
                           self.transform_2d)

    @nose.tools.raises(Exception)
    def test_save_directory(self):
        # Create a mock transform2d object
        # save() requires a second argument
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.save(directory_name, self.transform_2d)
