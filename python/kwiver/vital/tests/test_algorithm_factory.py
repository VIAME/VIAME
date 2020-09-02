"""
ckwg +29
Copyright 2019 by Kitware, Inc.
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

from __future__ import print_function, absolute_import

import nose.tools as nt

from kwiver.vital.algo import algorithm_factory, ImageObjectDetector


class TestImageObjectDetector(ImageObjectDetector):
    def __init__(self):
        ImageObjectDetector.__init__(self)


def test_algorithm_factory():

    # Add algorithm
    algorithm_factory.add_algorithm(
        "TestImageObjectDetector", "test detector", TestImageObjectDetector
    )

    nt.ok_(
        algorithm_factory.has_algorithm_impl_name(
            "image_object_detector", "TestImageObjectDetector"
        ),
        "TestImageObjectDetector not found by the factory",
    )

    # Check with an empty implementation
    nt.assert_equal(
        algorithm_factory.has_algorithm_impl_name("image_object_detector", ""), False
    )
    # Check with dummy implementation
    nt.assert_equal(
        algorithm_factory.has_algorithm_impl_name(
            "image_object_detector", "NotAnObjectDetector"
        ),
        False,
    )

    # Check that a registered implementation is returned by implementations
    nt.ok_(
        "TestImageObjectDetector"
        in algorithm_factory.implementations("image_object_detector"),
        "Dummy example_detector not registered",
    )
    # Check with an empty algorithm return empty list
    nt.assert_equal(len(algorithm_factory.implementations("")), 0)
    # Check with dummy algorithm returns empty list
    nt.assert_equal(len(algorithm_factory.implementations("NotAnAlgorithm")), 0)

    # Make sure creating works
    alg_out = algorithm_factory.create_algorithm(
        "image_object_detector", "TestImageObjectDetector"
    )

    nt.ok_(isinstance(alg_out, TestImageObjectDetector))
    nt.ok_(isinstance(alg_out, ImageObjectDetector))
