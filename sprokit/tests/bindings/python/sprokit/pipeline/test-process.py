#!/usr/bin/env python
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


def test_flags_as_set():
    from sprokit.pipeline import process

    # TODO: Make tests more rigorous (check more than just len()).

    a = process.PortFlags()

    # adding to the set
    a.add(process.PythonProcess.flag_required)
    a.add(process.PythonProcess.flag_input_mutable)
    a.add(process.PythonProcess.flag_input_nodep)
    a.add(process.PythonProcess.flag_input_static)

    # length
    if not len(a) == 4:
        test_error("len() does not work: expected 4, got %d" % len(a))

    # adding duplicate values
    a.add(process.PythonProcess.flag_required)

    if not len(a) == 4:
        test_error(".add() added a duplicate item: expected 4, got %d" % len(a))

    # adding invalid objects
    expect_exception('adding a value of an invalid type', TypeError,
                     process.PortFlags.add, a, True),

    # indexing failures
    expect_exception('getting an item by index', TypeError,
                     process.PortFlags.__getitem__, a, 0)
    expect_exception('deleting an item by index', TypeError,
                     process.PortFlags.__delitem__, a, 0)
    expect_exception('setting an item by index', TypeError,
                     process.PortFlags.__setitem__, a, 0, process.PythonProcess.flag_input_mutable)

    # 'in' keyword
    if process.PythonProcess.flag_required not in a:
        test_error("a value in the set is 'not in' the set")
    if process.PythonProcess.flag_output_const in a:
        test_error("a value not in the set is 'in' the set")

    # iteration
    for value in a:
        pass

    # boolean casting
    if not a:
        test_error("a non-empty set is False-like")

    b = process.PortFlags()

    if b:
        test_error("an empty set is True-like")

    # removal
    expect_exception('.pop() on an empty set', KeyError,
                     process.PortFlags.pop, b)
    expect_exception('.remove() with an item that does not exist in the set', KeyError,
                     process.PortFlags.remove, a, process.PythonProcess.flag_output_const)
    a.discard(process.PythonProcess.flag_output_const)

    if not len(a) == 4:
        test_error(".discard() removed an item not in the set")

    a.discard(process.PythonProcess.flag_input_static)

    if not len(a) == 3:
        test_error(".discard() did not remove an item from the set")

    a.remove(process.PythonProcess.flag_input_nodep)

    if not len(a) == 2:
        test_error(".remove() did not remove an item from the set")

    a.pop()

    if not len(a) == 1:
        test_error(".pop() did not remove an item from the set")

    a.clear()

    if a:
        test_error(".clear() did not make a False-like set")

    # copy
    b.add(process.PythonProcess.flag_required)

    c = b.copy()

    b.clear()

    if not c:
        test_error(".clear() on a set modified a set created using .copy()")

    c = b.copy()

    b.add(process.PythonProcess.flag_required)

    if c:
        test_error(".add() on a set modified a set created using .copy()")

    # set vs. set queries
    a.add(process.PythonProcess.flag_input_nodep)
    a.add(process.PythonProcess.flag_input_static)

    if not b.isdisjoint(a):
        test_error(".isdisjoint() does not work")
    if b.issubset(a):
        test_error(".issubset() does not work")
    if a.issuperset(b):
        test_error(".issuperset() does not work")

    a.add(process.PythonProcess.flag_required)

    if b.isdisjoint(a):
        test_error(".isdisjoint() does not work")
    if not b.issubset(a):
        test_error(".issubset() does not work")
    if not a.issuperset(b):
        test_error(".issuperset() does not work")

    u = a.union(b)

    if not len(u) == 3:
        test_error(".union() does not work: expected 3, got %d" % len(u))

    d = a.difference(b)

    if not len(d) == 2:
        test_error(".difference() does not work: expected 2, got %d" % len(d))

    i = a.intersection(b)

    if not len(i) == 1:
        test_error(".intersection() does not work: expected 1, got %d" % len(i))

    b.add(process.PythonProcess.flag_output_const)

    s = a.symmetric_difference(b)

    if not len(s) == 3:
        test_error(".symmetric_difference() does not work: expected 3, got %d" % len(s))

    a.update(b)

    if not len(a) == 4:
        test_error(".update() does not work: expected 4, got %d" % len(a))


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments( test-name, data-dir, path")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from sprokit.test.test import *

    run_test(testname, find_tests(locals()))
