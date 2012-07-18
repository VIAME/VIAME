#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.config
        import vistk.pipeline.process
        import vistk.pipeline.process_cluster
    except:
        test_error("Failed to import the process_cluster module")


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import process
    from vistk.pipeline import process_cluster

    process_cluster.PythonProcessCluster.type_any
    process_cluster.PythonProcessCluster.type_none
    process_cluster.PythonProcessCluster.type_data_dependent
    process_cluster.PythonProcessCluster.type_flow_dependent
    process_cluster.PythonProcessCluster.flag_output_const
    process_cluster.PythonProcessCluster.flag_input_static
    process_cluster.PythonProcessCluster.flag_input_mutable
    process_cluster.PythonProcessCluster.flag_required

    class BaseProcess(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

    c = config.empty_config()

    p = BaseProcess(c)

    if process_cluster.cluster_from_process(p) is not None:
        test_error("A non-cluster process was detected as a cluster process")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'api_calls':
        test_api_calls()
    else:
        test_error("No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from vistk.test.test import *

    try:
        main(testname)
    except BaseException as e:
        test_error("Unexpected exception: %s" % str(e))
