#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.pipeline
    except:
        test_error("Failed to import the pipeline module")


def test_create():
    from vistk.pipeline import config
    from vistk.pipeline import pipeline

    c = config.empty_config()

    pipeline.Pipeline()
    pipeline.Pipeline(c)


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import edge
    from vistk.pipeline import modules
    from vistk.pipeline import pipeline
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    p = pipeline.Pipeline()

    proc_type1 = 'numbers'
    proc_type2 = 'print_number'
    proc_type3 = 'orphan_cluster'

    proc_name1 = 'src'
    proc_name2 = 'snk'
    proc_name3 = 'orp'

    port_name1 = 'number'
    port_name2 = 'number'

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    proc1 = reg.create_process(proc_type1, proc_name1)

    conf_name = 'output'

    c = config.empty_config()

    c.set_value(conf_name, 'test-python-pipeline-api_calls-print_number.txt')
    proc2 = reg.create_process(proc_type2, proc_name2, c)

    proc3 = reg.create_process(proc_type3, proc_name3)

    p.add_process(proc1)
    p.add_process(proc2)
    p.add_process(proc3)
    p.connect(proc_name1, port_name1,
              proc_name2, port_name2)
    p.process_names()
    p.process_by_name(proc_name1)
    p.cluster_names()
    p.cluster_by_name(proc_name3)
    p.connections_from_addr(proc_name1, port_name1)
    p.connection_to_addr(proc_name2, port_name2)

    p.disconnect(proc_name1, port_name1,
                 proc_name2, port_name2)
    p.remove_process(proc_name1)
    p.remove_process(proc_name3)

    # Restore the pipeline so that setup_pipeline works.
    p.add_process(proc1)
    p.connect(proc_name1, port_name1,
              proc_name2, port_name2)

    p.setup_pipeline()

    p.upstream_for_process(proc_name2)
    p.upstream_for_port(proc_name2, port_name2)
    p.downstream_for_process(proc_name1)
    p.downstream_for_port(proc_name1, port_name1)
    p.sender_for_port(proc_name2, port_name2)
    p.receivers_for_port(proc_name1, port_name1)
    p.edge_for_connection(proc_name1, port_name1,
                          proc_name2, port_name2)
    p.input_edges_for_process(proc_name2)
    p.input_edge_for_port(proc_name2, port_name2)
    p.output_edges_for_process(proc_name1)
    p.output_edges_for_port(proc_name1, port_name1)

    p.is_setup()
    p.setup_successful()

    c = config.empty_config()

    p.reconfigure(c)

    p.reset()


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
