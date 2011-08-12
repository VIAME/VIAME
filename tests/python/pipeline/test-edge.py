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

    e = edge.Edge(c)


def test_datum_create():
    from vistk.pipeline import datum
    from vistk.pipeline import edge
    from vistk.pipeline import stamp

    d = datum.complete()
    s = stamp.new_stamp()

    edge.EdgeDatum(d, s)


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'datum_create':
        test_datum_create()
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
