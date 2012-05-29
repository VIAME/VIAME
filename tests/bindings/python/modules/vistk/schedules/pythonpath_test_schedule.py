#ckwg +5
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.pipeline import config
from vistk.pipeline import pipeline
from vistk.pipeline import schedule


class TestPythonSchedule(schedule.PythonSchedule):
    def __init__(self, conf, pipe):
        schedule.PythonSchedule.__init__(self, conf, pipe)


def __vistk_register__():
    from vistk.pipeline import schedule_registry

    module_name = 'python:test.pythonpath.test'

    reg = schedule_registry.ScheduleRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_schedule('pythonpath_test_schedule', 'A test schedule.', TestPythonSchedule)

    reg.mark_module_as_loaded(module_name)
