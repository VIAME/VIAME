#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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

    pipeline.Pipeline(c)


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import edge
    from vistk.pipeline import modules
    from vistk.pipeline import pipeline
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    c = config.empty_config()

    p = pipeline.Pipeline(c)

    proc_type1 = 'numbers'
    proc_type2 = 'print_number'

    proc_name1 = 'src'
    proc_name2 = 'snk'

    port_name1 = 'number'
    port_name2 = 'number'

    group_name = 'group'
    group_iport = 'iport'
    group_oport = 'oport'

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    c.set_value(process_registry.Process.config_name, proc_name1)
    proc1 = reg.create_process(proc_type1, c)

    conf_name = 'output'

    c.set_value(process_registry.Process.config_name, proc_name2)
    c.set_value(conf_name, 'test-python-pipeline-api_calls-print_number.txt')
    proc2 = reg.create_process(proc_type2, c)

    p.add_process(proc1)
    p.add_process(proc2)
    p.add_group(group_name)
    p.connect(proc_name1, port_name1,
              proc_name2, port_name2)
    p.map_input_port(group_name, group_iport,
                     proc_name2, port_name2,
                     process.PortFlags())
    p.map_output_port(group_name, group_oport,
                      proc_name1, port_name1,
                      process.PortFlags())
    p.process_names()
    p.process_by_name(proc_name1)
    p.connections_from_addr(proc_name1, port_name1)
    p.connection_to_addr(proc_name2, port_name2)
    p.groups()
    p.input_ports_for_group(group_name)
    p.output_ports_for_group(group_name)
    p.mapped_group_input_port_flags(group_name, group_iport)
    p.mapped_group_output_port_flags(group_name, group_oport)
    p.mapped_group_input_ports(group_name, group_iport)
    p.mapped_group_output_port(group_name, group_oport)

    p.disconnect(proc_name1, port_name1,
                 proc_name2, port_name2)
    p.unmap_input_port(group_name, group_iport,
                       proc_name2, port_name2)
    p.unmap_output_port(group_name, group_oport,
                        proc_name1, port_name1)
    p.remove_process(proc_name1)
    p.remove_group(group_name)

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

    p.start()
    p.stop()

    p.reset()


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
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
