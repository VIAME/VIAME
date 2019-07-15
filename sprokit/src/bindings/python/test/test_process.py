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

Tests for sprokit process
"""

from __future__ import print_function, absolute_import
import nose.tools

from sprokit.pipeline import process
from sprokit.pipeline import datum, DatumType
from sprokit.pipeline import config
from sprokit.pipeline import edge
from sprokit.pipeline import stamp

class TestSprokitProces( object ):

    def test_new(self):
        """
        Test process creation with empty configuration
        """
        cfg = config.empty_config()
        proc = process.PythonProcess(cfg)

    def test_peek_at_datum_on_port(self):
        """
        Test peek at datum on a test port with complete datum
        """
        cfg = config.empty_config()
        # Create Dummy Receiver process
        receiver_proc = process.PythonProcess(cfg)
        optional = process.PortFlags()
        receiver_proc.declare_input_port("test_port", "test", optional, "test_port")
        # Create an Edge and connect input port to the edge
        test_edge = edge.Edge()
        receiver_proc.connect_input_port("test_port", test_edge)
        # Create an Edge Datum and push it to the port
        s = stamp.new_stamp(1)
        e_datum = edge.EdgeDatum(datum.complete(), s)
        test_edge.push_datum(e_datum)
        nose.tools.assert_equal( receiver_proc.peek_at_datum_on_port("test_port").type(),
                                 DatumType.complete)
