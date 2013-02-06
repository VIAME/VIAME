#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import(path):
    try:
        import vistk.pipeline_util.bake
    except:
        test_error("Failed to import the bake module")


def test_simple_pipeline(path):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import modules
    from vistk.pipeline_util import bake
    from vistk.pipeline_util import load

    blocks = load.load_pipe_file(path)

    modules.load_known_modules()

    bake.bake_pipe_file(path)
    with open(path, 'r') as fin:
        bake.bake_pipe(fin)
    bake.bake_pipe_blocks(blocks)
    bake.extract_configuration(blocks)


def test_cluster_multiplier(path):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import modules
    from vistk.pipeline_util import bake
    from vistk.pipeline_util import load

    blocks = load.load_cluster_file(path)

    modules.load_known_modules()

    bake.bake_cluster_file(path)
    with open(path, 'r') as fin:
        bake.bake_cluster(fin)
    info = bake.bake_cluster_blocks(blocks)

    conf = config.empty_config()

    info.type()
    info.description()
    info.create()
    info.create(conf)

    bake.register_cluster(info)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 5:
        test_error("Expected four arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    pipeline_dir = sys.argv[4]

    tests = \
        { 'import': test_import
        , 'simple_pipeline': test_simple_pipeline
        , 'cluster_multiplier': test_cluster_multiplier
        }

    path = os.path.join(pipeline_dir, '%s.pipe' % testname)

    from vistk.test.test import *

    run_test(testname, tests, path)
