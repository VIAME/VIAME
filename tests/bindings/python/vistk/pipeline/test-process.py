#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.process
    except:
        test_error("Failed to import the process module")


def test_create():
    from vistk.pipeline import datum
    from vistk.pipeline import process

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
    process.ConfInfo('default', 'desc')
    process.DataInfo(True, datum.DatumType.invalid)
    process.DataCheck.none
    process.DataCheck.sync
    process.DataCheck.valid


def test_api_calls():
    from vistk.pipeline import datum
    from vistk.pipeline import process

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

    a = process.ConfInfo('default', 'desc')
    a.default
    a.description

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

    from vistk.test.test import *

    run_test(testname, tests)
