#!/usr/bin/env python
#ckwg +28
# Copyright 2011-2013 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def test_import():
    try:
        from sprokit.pipeline import config
        import sprokit.pipeline.scheduler_factory
    except:
        test_error("Failed to import the scheduler_factory module")


def test_create():
    from sprokit.pipeline import config
    from sprokit.pipeline import scheduler_factory

    scheduler_factory.SchedulerType()
    ## scheduler_factory.SchedulerTypes()
    scheduler_factory.SchedulerDescription()
    scheduler_factory.SchedulerModule()


def test_api_calls():
    from sprokit.pipeline import config
    from sprokit.pipeline import modules
    from sprokit.pipeline import pipeline
    from sprokit.pipeline import scheduler_factory

    modules.load_known_modules()

    sched_type = 'thread_per_process'
    c = config.empty_config()
    p = pipeline.Pipeline()

    scheduler_factory.create_scheduler(sched_type, p)
    scheduler_factory.create_scheduler(sched_type, p, c)
    scheduler_factory.types()
    scheduler_factory.description(sched_type)
    scheduler_factory.default_type


def example_scheduler(check_init):
    from sprokit.pipeline import scheduler

    class PythonExample(scheduler.PythonScheduler):
        def __init__(self, pipe, conf):
            scheduler.PythonScheduler.__init__(self, pipe, conf)

            self.ran_start = check_init
            self.ran_wait = check_init
            self.ran_stop = check_init
            self.ran_pause = check_init
            self.ran_resume = check_init

        def _start(self):
            self.ran_start = True

        def _wait(self):
            self.ran_wait = True

        def _stop(self):
            self.ran_stop = True

        def _pause(self):
            self.ran_pause = True

        def _resume(self):
            self.ran_resume = True

        def __del__(self):
            if not self.ran_start:
                test_error("start override was not called")
            if not self.ran_wait:
                test_error("wait override was not called")
            if not self.ran_stop:
                test_error("stop override was not called")
            if not self.ran_pause:
                test_error("pause override was not called")
            if not self.ran_resume:
                test_error("resume override was not called")

    return PythonExample


def test_register():
    from sprokit.pipeline import config
    from sprokit.pipeline import modules
    from sprokit.pipeline import pipeline
    from sprokit.pipeline import scheduler_factory

    modules.load_known_modules()

    sched_type = 'python_example'
    sched_desc = 'simple description'

    scheduler_factory.add_scheduler(sched_type, sched_desc, example_scheduler(True))

    if not sched_desc == scheduler_factory.description(sched_type):
        test_error("Description was not preserved when registering")

    p = pipeline.Pipeline()

    try:
        s = scheduler_factory.create_scheduler(sched_type, p)
        if s is None:
            raise Exception()
    except:
        test_error("Could not create newly registered scheduler type")


def test_wrapper_api():
    from sprokit.pipeline import config
    from sprokit.pipeline import modules
    from sprokit.pipeline import pipeline
    from sprokit.pipeline import process_factory
    from sprokit.pipeline import scheduler_factory

    sched_type = 'python_example'
    sched_desc = 'simple description'

    modules.load_known_modules()

    scheduler_factory.add_scheduler(sched_type, sched_desc, example_scheduler(False))

    p = pipeline.Pipeline()

    proc_type = 'orphan'
    proc_name = 'orphan'

    proc = process_factory.create_process(proc_type, proc_name)

    p.add_process(proc)

    def check_scheduler(s):
        if s is None:
            test_error("Got a 'None' scheduler")
            return

        s.start()
        s.pause()
        s.resume()
        s.stop()
        s.start()
        s.wait()

        del s

    p.reset()
    p.setup_pipeline()

    s = scheduler_factory.create_scheduler(sched_type, p)
    check_scheduler(s)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from sprokit.test.test import *

    run_test(testname, find_tests(locals()))
