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
        import vistk.pipeline.schedule_registry
    except:
        log("Error: Failed to import the schedule_registry module")


def test_create():
    from vistk.pipeline import schedule_registry

    schedule_registry.ScheduleRegistry.self()
    schedule_registry.ScheduleType()
    schedule_registry.ScheduleTypes()
    schedule_registry.ScheduleDescription()


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import pipeline
    from vistk.pipeline import schedule_registry

    modules.load_known_modules()

    reg = schedule_registry.ScheduleRegistry.self()

    sched_type = 'thread_per_process'
    c = config.empty_config()
    p = pipeline.Pipeline(c)

    reg.create_schedule(sched_type, c, p)
    reg.types()
    reg.description(sched_type)
    reg.default_type


def example_schedule():
    from vistk.pipeline import schedule

    class PythonExample(schedule.PythonSchedule):
        def __init__(self, conf, pipe):
            schedule.PythonSchedule.__init__(self, conf, pipe)

            self.ran_start = False
            self.ran_wait = False
            self.ran_stop = False

        def start(self):
            self.ran_start = True

        def wait(self):
            self.ran_wait = True

        def stop(self):
            self.ran_stop = True

        def check(self):
            if not self.ran_start:
                log("Error: start override was not called")
            if not self.ran_wait:
                log("Error: wait override was not called")
            if not self.ran_stop:
                log("Error: stop override was not called")

    return PythonExample


def test_register():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import pipeline
    from vistk.pipeline import schedule_registry

    modules.load_known_modules()

    reg = schedule_registry.ScheduleRegistry.self()

    sched_type = 'python_example'
    sched_desc = 'simple description'

    reg.register_schedule(sched_type, sched_desc, example_schedule())

    if not sched_desc == reg.description(sched_type):
        log("Error: Description was not preserved when registering")

    c = config.empty_config()
    p = pipeline.Pipeline(c)

    try:
        s = reg.create_schedule(sched_type, c, p)
        if s is None:
            raise Exception()
    except:
        log("Error: Could not create newly registered schedule type")


def test_wrapper_api():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import pipeline
    from vistk.pipeline import schedule_registry

    sched_type = 'python_example'
    sched_desc = 'simple description'

    reg = schedule_registry.ScheduleRegistry.self()

    reg.register_schedule(sched_type, sched_desc, example_schedule())

    c = config.empty_config()
    p = pipeline.Pipeline(c)

    def check_schedule(s):
        if s is None:
            log("Error: Got a 'None' schedule")
            return

        s.start()
        s.wait()
        s.stop()

        s.check()

    s = reg.create_schedule(sched_type, c, p)
    check_schedule(s)


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
    elif testname == 'register':
        test_register()
    elif testname == 'wrapper_api':
        test_wrapper_api()
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
