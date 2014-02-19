#!@PYTHON_EXECUTABLE@
#ckwg +28
# Copyright 2011-2013 by Kitware, Inc.
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


def test_import():
    try:
        import sprokit.pipeline.process
    except:
        test_error("Failed to import the process module")


def test_create():
    from sprokit.pipeline import datum
    from sprokit.pipeline import process

    process.ProcessType()
    process.ProcessTypes()
    process.ProcessName()
    process.ProcessNames()
    process.ProcessProperty()
    process.ProcessProperties()
    process.PortDescription()
    process.PortFrequency(1)
    process.PortFrequency(1, 1)
    process.Port()
    process.Ports()
    process.PortType()
    process.PortFlag()
    process.PortFlags()
    process.PortAddr()
    process.PortAddrs()
    process.Connection()
    process.Connections()
    process.PortInfo('type', process.PortFlags(), 'desc', process.PortFrequency(1, 1))
    process.ConfInfo('default', 'desc', False)
    process.DataInfo(True, datum.DatumType.invalid)
    process.DataCheck.none
    process.DataCheck.sync
    process.DataCheck.valid


def test_api_calls():
    from sprokit.pipeline import datum
    from sprokit.pipeline import process

    a = process.PortAddr()
    a.process
    a.port
    a.process = ''
    a.port = ''

    f = process.PortFrequency(1, 1)

    a = process.PortInfo('type', process.PortFlags(), 'desc', f)
    a.type
    a.flags
    a.description
    a.frequency

    a = process.ConfInfo('default', 'desc', False)
    a.default
    a.description
    a.tunable

    a = process.DataInfo(True, datum.DatumType.invalid)
    a.in_sync
    a.max_status

    process.PythonProcess.property_no_threads
    process.PythonProcess.property_no_reentrancy
    process.PythonProcess.property_unsync_input
    process.PythonProcess.property_unsync_output
    process.PythonProcess.port_heartbeat
    process.PythonProcess.config_name
    process.PythonProcess.config_type
    process.PythonProcess.type_any
    process.PythonProcess.type_none
    process.PythonProcess.type_data_dependent
    process.PythonProcess.type_flow_dependent
    process.PythonProcess.flag_output_const
    process.PythonProcess.flag_output_shared
    process.PythonProcess.flag_input_static
    process.PythonProcess.flag_input_mutable
    process.PythonProcess.flag_input_nodep
    process.PythonProcess.flag_required


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    tests = \
        { 'import': test_import
        , 'create': test_create
        , 'api_calls': test_api_calls
        }

    from sprokit.test.test import *

    run_test(testname, tests)
