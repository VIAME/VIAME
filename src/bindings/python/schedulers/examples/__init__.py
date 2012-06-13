#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.schedulers.examples import pythread_per_process_scheduler


def __vistk_register__():
    from vistk.pipeline import scheduler_registry

    module_name = 'python:schedulers.examples'

    reg = scheduler_registry.SchedulerRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_scheduler('pythread_per_process', 'Run each process in its own Python thread', pythread_per_process_scheduler.PyThreadPerProcessScheduler)

    reg.mark_module_as_loaded(module_name)
