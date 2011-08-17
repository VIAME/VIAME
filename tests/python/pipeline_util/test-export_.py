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
        import vistk.pipeline_util.export_
    except:
        log("Error: Failed to import the export_ module")


def test_simple_pipeline(path):
    import os

    from vistk.pipeline import pipeline
    from vistk.pipeline import modules
    from vistk.pipeline_util import bake
    from vistk.pipeline_util import export_

    modules.load_known_modules()

    p = bake.bake_pipe_file(path)
    r, w = os.pipe()

    name = 'graph'

    export_.export_dot(w, p, name)

    os.close(r)
    os.close(w)


def main(testname, path):
    if testname == 'import':
        test_import()
    elif testname == 'simple_pipeline':
        test_simple_pipeline(path)
    else:
        log("Error: No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 5:
        log("Error: Expected four arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    pipeline_dir = sys.argv[4]

    path = os.path.join(pipeline_dir, '%s.pipe' % testname)

    main(testname, path)
