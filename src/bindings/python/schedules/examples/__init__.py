#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.schedules.examples import pythread_per_process_schedule


def __vistk_register__():
    from vistk.pipeline import schedule_registry

    module_name = 'python:schedules.examples'

    reg = schedule_registry.ScheduleRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_schedule('pythread_per_process', 'Runs each procss in its own Python thread', pythread_per_process_schedule.PyThreadPerProcessSchedule)

    reg.mark_module_as_loaded(module_name)
