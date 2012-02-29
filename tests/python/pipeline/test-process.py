#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def test_import():
    try:
        import vistk.pipeline.process
    except:
        log("Error: Failed to import the process module")


def test_create():
    from vistk.pipeline import datum
    from vistk.pipeline import process

    process.ProcessName()
    process.ProcessNames()
    process.ProcessConstraint()
    process.ProcessConstraints()
    process.PortDescription()
    process.Port()
    process.Ports()
    process.PortType()
    process.PortFlag()
    process.PortFlags()
    process.PortAddr()
    process.PortAddrs()
    process.PortInfo('type', process.PortFlags(), 'desc')
    process.ConfInfo('default', 'desc')
    process.DataInfo(True, True, datum.DatumType.invalid)


def test_api_calls():
    from vistk.pipeline import datum
    from vistk.pipeline import process

    a = process.PortAddr()
    a.process
    a.port
    a.process = ''
    a.port = ''

    a = process.PortInfo('type', process.PortFlags(), 'desc')
    a.type
    a.flags
    a.description

    a = process.ConfInfo('default', 'desc')
    a.default
    a.description

    a = process.DataInfo(True, True, datum.DatumType.invalid)
    a.same_color
    a.in_sync
    a.max_status

    process.PythonProcess.constraint_no_threads
    process.PythonProcess.constraint_python
    process.PythonProcess.constraint_no_reentrancy
    process.PythonProcess.constraint_unsync_input
    process.PythonProcess.constraint_unsync_output
    process.PythonProcess.port_heartbeat
    process.PythonProcess.config_name
    process.PythonProcess.config_type
    process.PythonProcess.type_any
    process.PythonProcess.type_none
    process.PythonProcess.type_data_dependent
    process.PythonProcess.type_flow_dependent
    process.PythonProcess.flag_output_const
    process.PythonProcess.flag_input_mutable
    process.PythonProcess.flag_input_nodep
    process.PythonProcess.flag_required


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
    else:
        log("Error: No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        log("Error: Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    try:
        main(testname)
    except BaseException as e:
        log("Error: Unexpected exception: %s" % str(e))
