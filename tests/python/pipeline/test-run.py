#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def ensure_exception(action, func, *args):
    got_exception = False

    try:
        func(*args)
    except:
        got_exception = True

    if not got_exception:
        log("Error: Did not get exception when %s" % action)


def make_source(conf):
    from vistk.pipeline import process

    class Source(process.PythonProcess):
        def __init__(self, conf):
            from vistk.pipeline_types import basic

            process.PythonProcess.__init__(self, conf)

            self.conf_start = 'start'
            self.conf_end = 'end'

            info = process.ConfInfo(str(0), 'Starting number')

            self.declare_configuration_key(self.conf_start, info)

            info = process.ConfInfo(str(10), 'Ending number')

            self.declare_configuration_key(self.conf_end, info)

            self.port_color = 'color'
            self.port_output = 'number'

            info = process.PortInfo(self.type_none, process.PortFlags(), 'color port')

            self.declare_input_port(self.port_color, info)

            required = process.PortFlags()
            required.add(self.flag_required)
            info = process.PortInfo(basic.t_integer, required, 'output port')

            self.declare_output_port(self.port_output, info)

        def _init(self):
            from vistk.pipeline import stamp

            self.counter = int(self.config_value(self.conf_start))
            self.end = int(self.config_value(self.conf_end))

            self.has_color = False
            if self.input_port_edge(self.port_color):
                self.has_color = True

            self.stamp = self.heartbeat_stamp()

        def _step(self):
            from vistk.pipeline import datum
            from vistk.pipeline import edge
            from vistk.pipeline import stamp

            complete = False

            if self.counter >= self.end:
                complete = True
            else:
                dat = datum.new(self.counter)
                self.counter += 1

            if self.has_color:
                color_dat = self.grab_from_port(self.port_color)

                color_status = color_dat.datum.type()

                if color_status == datum.DatumType.complete:
                    complete = True
                elif color_status == datum.DatumType.error:
                    dat = datum.error('Error on the color edge.')
                elif color_status == datum.DatumType.invalid:
                    dat = datum.error('Invalid status on the color edge.')

                self.stamp = stamp.recolored_stamp(self.stamp, color_dat.stamp)

            if complete:
                self.mark_process_as_complete()
                dat = datum.complete()

            edat = edge.EdgeDatum(dat, self.stamp)

            self.push_to_port(self.port_output, edat)

            self.stamp = stamp.incremented_stamp(self.stamp)

            self._base_step()

    return Source(conf)


def make_sink(conf):
    from vistk.pipeline import process

    class Sink(process.PythonProcess):
        def __init__(self, conf):
            from vistk.pipeline_types import basic

            process.PythonProcess.__init__(self, conf)

            self.conf_output = 'output'

            info = process.ConfInfo('output.txt', 'Output file name')

            self.declare_configuration_key(self.conf_output, info)

            self.port_input = 'number'

            required = process.PortFlags()
            required.add(self.flag_required)
            info = process.PortInfo(basic.t_integer, required, 'input port')

            self.declare_input_port(self.port_input, info)

        def _init(self):
            output = self.config_value(self.conf_output)

            self.fout = open(output, 'w+')

            self._base_init()

        def _step(self):
            from vistk.pipeline import datum

            dat = self.grab_datum_from_port(self.port_input)
            num = dat.get_datum()

            self.fout.write('%d\n' % num)
            self.fout.flush()

            self._base_step()

    return Sink(conf)


def create_process(type, conf):
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    p = reg.create_process(type, conf)

    return p


def run_pipeline(sched_type, conf, pipe):
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import schedule_registry

    modules.load_known_modules()

    reg = schedule_registry.ScheduleRegistry.self()

    s = reg.create_schedule(sched_type, conf, pipe)

    s.start()
    s.wait()


def check_file(fname, expect):
    with open(fname, 'r') as fin:
        ints = map(lambda l: int(l.strip()), list(fin))

        num_ints = len(ints)
        num_expect = len(expect)

        if not num_ints == num_expect:
            log("Error: Got %d results when %d were expected." % (num_ints, num_expect))

        res = zip(ints, expect)

        line = 1

        for i, e in res:
            if not i == e:
                log("Error: Result %d is %d, where %d was expected" % (line, i, e))
            line += 1


def test_python_to_python(sched_type):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import process

    name_source = 'source'
    name_sink = 'sink'

    port_output = 'number'
    port_input = 'number'

    min = 0
    max = 10
    output_file = 'test-python-run-python_to_python.txt'

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source)
    c.set_value('start', str(min))
    c.set_value('end', str(max))

    s = make_source(c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_sink)
    c.set_value('output', output_file)

    t = make_sink(c)

    p = pipeline.Pipeline(c)

    p.add_process(s)
    p.add_process(t)

    p.connect(name_source, port_output,
              name_sink, port_input)

    p.setup_pipeline()

    run_pipeline(sched_type, c, p)

    check_file(output_file, range(min, max))


def test_cpp_to_python(sched_type):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import process

    name_source = 'source'
    name_sink = 'sink'

    port_output = 'number'
    port_input = 'number'

    min = 0
    max = 10
    output_file = 'test-python-run-cpp_to_python.txt'

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source)
    c.set_value('start', str(min))
    c.set_value('end', str(max))

    s = create_process('numbers', c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_sink)
    c.set_value('output', output_file)

    t = make_sink(c)

    p = pipeline.Pipeline(c)

    p.add_process(s)
    p.add_process(t)

    p.connect(name_source, port_output,
              name_sink, port_input)

    p.setup_pipeline()

    run_pipeline(sched_type, c, p)

    check_file(output_file, range(min, max))


def test_python_to_cpp(sched_type):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import process

    name_source = 'source'
    name_sink = 'sink'

    port_output = 'number'
    port_input = 'number'

    min = 0
    max = 10
    output_file = 'test-python-run-python_to_cpp.txt'

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source)
    c.set_value('start', str(min))
    c.set_value('end', str(max))

    s = make_source(c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_sink)
    c.set_value('output', output_file)

    t = create_process('print_number', c)

    p = pipeline.Pipeline(c)

    p.add_process(s)
    p.add_process(t)

    p.connect(name_source, port_output,
              name_sink, port_input)

    p.setup_pipeline()

    run_pipeline(sched_type, c, p)

    check_file(output_file, range(min, max))


def test_python_via_cpp(sched_type):
    from vistk.pipeline import config
    from vistk.pipeline import pipeline
    from vistk.pipeline import process

    name_source = 'source'
    name_source1 = 'source1'
    name_source2 = 'source2'
    name_mult = 'mult'
    name_sink = 'sink'

    port_color = 'color'
    port_output = 'number'
    port_factor1 = 'factor1'
    port_factor2 = 'factor2'
    port_product = 'product'
    port_input = 'number'

    min1 = 0
    max1 = 10
    min2 = 10
    max2 = 15
    output_file = 'test-python-run-python_via_cpp.txt'

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source)

    s = create_process('source', c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source1)
    c.set_value('start', str(min1))
    c.set_value('end', str(max1))

    s1 = make_source(c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_source2)
    c.set_value('start', str(min2))
    c.set_value('end', str(max2))

    s2 = make_source(c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_mult)

    m = create_process('multiplication', c)

    c = config.empty_config()

    c.set_value(process.PythonProcess.config_name, name_sink)
    c.set_value('output', output_file)

    t = make_sink(c)

    p = pipeline.Pipeline(c)

    p.add_process(s)
    p.add_process(s1)
    p.add_process(s2)
    p.add_process(m)
    p.add_process(t)

    p.connect(name_source, port_color,
              name_source1, port_color)
    p.connect(name_source, port_color,
              name_source2, port_color)
    p.connect(name_source1, port_output,
              name_mult, port_factor1)
    p.connect(name_source2, port_output,
              name_mult, port_factor2)
    p.connect(name_mult, port_product,
              name_sink, port_input)

    p.setup_pipeline()

    run_pipeline(sched_type, c, p)

    check_file(output_file, [a * b for a, b in zip(range(min1, max1), range(min2, max2))])


def main(testname, sched_type):
    if testname == 'python_to_python':
        test_python_to_python(sched_type)
    elif testname == 'cpp_to_python':
        test_cpp_to_python(sched_type)
    elif testname == 'python_to_cpp':
        test_python_to_cpp(sched_type)
    elif testname == 'python_via_cpp':
        test_python_via_cpp(sched_type)
    else:
        log("Error: No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        log("Error: Expected three arguments")
        sys.exit(1)

    (testname, sched_type) = tuple(sys.argv[1].split('-', 1))

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    try:
        main(testname, sched_type)
    except BaseException as e:
        log("Error: Unexpected exception: %s" % str(e))
