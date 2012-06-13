#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.pipeline import config
from vistk.pipeline import pipeline
from vistk.pipeline import scheduler


class TestPythonScheduler(scheduler.PythonScheduler):
    def __init__(self, conf, pipe):
        scheduler.PythonScheduler.__init__(self, conf, pipe)


def __vistk_register__():
    from vistk.pipeline import scheduler_registry

    module_name = 'python:test.pythonpath.test'

    reg = scheduler_registry.SchedulerRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_scheduler('pythonpath_test_scheduler', 'A test scheduler.', TestPythonScheduler)

    reg.mark_module_as_loaded(module_name)
