"""
ckwg +31
Copyright 2020 by Kitware, Inc.
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

==============================================================================

Tests for Python interface to vital::iqr_feedback

"""

from kwiver.vital.types import IQRFeedback, UID

import nose.tools as nt
import numpy.testing as npt


class TestVitalIQRFeedback(object):
    def test_new(self):
        IQRFeedback()

    def test_set_and_get_query_id(self):
        iqrf = IQRFeedback()

        # First check default
        nt.assert_equals(iqrf.query_id.value(), "")
        nt.assert_false(iqrf.query_id.is_valid())

        # Now check setting and getting a few values
        iqrf.query_id = UID("first")
        nt.assert_equals(iqrf.query_id.value(), "first")

        iqrf.query_id = UID("second")
        nt.assert_equals(iqrf.query_id.value(), "second")

        iqrf.query_id = UID("42")
        nt.assert_equals(iqrf.query_id.value(), "42")

        # Try setting back to empty
        iqrf.query_id = UID()
        nt.assert_equals(iqrf.query_id.value(), "")

    @nt.raises(TypeError)
    def test_bad_set_query_id(self):
        iqrf = IQRFeedback()
        iqrf.query_id = "string, not uid"

    def test_pos_and_neg_ids(self):
        iqrf = IQRFeedback()

        npt.assert_array_equal(iqrf.positive_ids, [])
        npt.assert_array_equal(iqrf.negative_ids, [])

        l1 =  [2, 5, 6, 7, 8]
        l2 =  [1, 3, 4 ]

        iqrf.positive_ids = l1
        iqrf.negative_ids = l2

        npt.assert_array_equal(iqrf.positive_ids, l1)
        npt.assert_array_equal(iqrf.negative_ids, l2)

        iqrf.positive_ids = l2
        iqrf.negative_ids = l1

        npt.assert_array_equal(iqrf.positive_ids, l2)
        npt.assert_array_equal(iqrf.negative_ids, l1)

        iqrf.positive_ids = []
        iqrf.negative_ids = []

        npt.assert_array_equal(iqrf.positive_ids, [])
        npt.assert_array_equal(iqrf.negative_ids, [])

    @nt.raises(TypeError)
    def test_bad_set_pos_ids(self):
        iqrf = IQRFeedback()
        iqrf.positive_ids = "string, not list"


    @nt.raises(TypeError)
    def test_bad_set_neg_ids(self):
        iqrf = IQRFeedback()
        iqrf.negative_ids = "string, not list"
