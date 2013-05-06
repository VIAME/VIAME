#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from sprokit.processes.test import examples


def __sprokit_register__():
    from sprokit.pipeline import process_registry

    module_name = 'python:test.examples'

    reg = process_registry.ProcessRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_process('test_python_process', 'A test Python process', examples.TestPythonProcess)
    reg.register_process('pyprint_number', 'A Python process which prints numbers', examples.PythonPrintNumberProcess)

    reg.mark_module_as_loaded(module_name)
