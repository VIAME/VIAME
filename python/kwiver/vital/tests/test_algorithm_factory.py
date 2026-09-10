"""
ckwg +29
Copyright 2019-2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Tests for python algorithm factory class
"""

from kwiver.vital import plugin_management
from kwiver.vital.algo import ImageObjectDetector


# an algorithm needs to implement a minimal pluggable API:
class DummyImageObjectDetector(ImageObjectDetector):
    def __init__(self):
        ImageObjectDetector.__init__(self)

    @classmethod
    def from_config(cls, c):
        return DummyImageObjectDetector()

    @classmethod
    def get_default_config(cls, c):
        return


def test_algorithm_factory():
    # algorithm is automatically added since it is defined in the same file.
    vpm = plugin_management.plugin_manager_instance()
    vpm.load_all_plugins()

    assert ImageObjectDetector.has_algorithm_impl_name(
        "DummyImageObjectDetector"
    ), "TestImageObjectDetector not found by the factory"

    # Check with an empty implementation
    assert ImageObjectDetector.has_algorithm_impl_name("") == False

    # Check with non existsting implementation
    assert ImageObjectDetector.has_algorithm_impl_name("NotAnObjectDetector") == False

    # Check that a registered implementation is returned by implementations
    assert (
        "DummyImageObjectDetector" in ImageObjectDetector.registered_names()
    ), "Dummy example_detector not registered"

    # Check with an empty algorithm return empty list
    assert len(vpm.impl_names("")) == 0
    # Check with dummy algorithm returns empty list
    assert len(vpm.impl_names("NotAnAlgorithm")) == 0

    # Make sure creating works
    alg_out = ImageObjectDetector.create_algorithm("DummyImageObjectDetector")

    assert isinstance(alg_out, DummyImageObjectDetector)
    assert isinstance(alg_out, ImageObjectDetector)
