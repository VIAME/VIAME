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
        import vistk.pipeline.edge
    except:
        log("Error: Failed to import the edge module")


def test_create():
    from vistk.pipeline import config
    from vistk.pipeline import edge

    c = config.empty_config()

    edge.Edge(c)
    edge.Edges()


def test_datum_create():
    from vistk.pipeline import datum
    from vistk.pipeline import edge
    from vistk.pipeline import stamp

    d = datum.complete()
    s = stamp.new_stamp()

    edge.EdgeDatum(d, s)
    edge.EdgeData()


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import datum
    from vistk.pipeline import edge
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry
    from vistk.pipeline import stamp

    c = config.empty_config()

    e = edge.Edge(c)

    e.makes_dependency()
    e.has_data()
    e.full_of_data()
    e.datum_count()

    d = datum.complete()
    s = stamp.new_stamp()

    ed = edge.EdgeDatum(d, s)

    e.push_datum(ed)
    e.get_datum()

    e.push_datum(ed)
    e.peek_datum()
    e.pop_datum()

    e.set_required_by_downstream(True)
    e.required_by_downstream()

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    p = reg.create_process('orphan', c)

    e.set_upstream_process(p)
    e.set_downstream_process(p)


def test_datum_api_calls():
    from vistk.pipeline import datum
    from vistk.pipeline import edge
    from vistk.pipeline import stamp

    d = datum.complete()
    s = stamp.new_stamp()

    ed = edge.EdgeDatum(d, s)

    ed.datum
    ed.datum = d
    ed.stamp
    ed.stamp = s


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'datum_create':
        test_datum_create()
    elif testname == 'api_calls':
        test_api_calls()
    elif testname == 'datum_api_calls':
        test_datum_api_calls()
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

    main(testname)
