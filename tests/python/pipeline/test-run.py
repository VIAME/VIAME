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


def create_process(type, conf):
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    p = reg.create_process(type, conf)

    return p


def run_pipeline(conf, pipe):
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import schedule_registry

    modules.load_known_modules()

    reg = schedule_registry.ScheduleRegistry.self()

    sched_type = 'sync'

    s = reg.create_schedule(sched_type, conf, pipe)

    s.start()
    s.wait()


def main(testname):
    #else:
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
