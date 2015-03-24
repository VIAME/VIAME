#ckwg +4
# Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

import sprokit.pipeline.process
import sprokit.pipeline.config
import sprokit.pipeline.process_registry
import os.path

class kw_print_number_process(sprokit.pipeline.process.PythonProcess):
    def __init__(self, conf ):
        sprokit.pipeline.process.PythonProcess.__init__(self, conf)

        # declare our configuration items
        self.declare_configuration_key(
            'output',
            '.',
            'The path for the output file.')

        # create port flags
        flags = sprokit.pipeline.process.PortFlags()
        flags.add(self.flag_required)

        # create input ports
        self.declare_input_port( # example
            'input',
            'integer',
            flags,
            'Where numbers are read from.')

    # ----------------------------------------------------------------
    def _configure(self):
        path = self.config_value('output')

        if not path:
            raise RuntimeError('The path given was empty')

        print "KEITH Path is ",path

        self.fout = open(path, 'w+')
        self.fout.flush()

        self._base_configure()

    # ----------------------------------------------------------------
    def _reset(self):
        self.fout.close()

        self._base_reset()

    def _step(self):
        num = self.grab_value_from_port('input')

        self.fout.write('%d\n' % num)

        self._base_step()

def __sprokit_register__():

    module_name = 'python:kwiver.print_number'

    reg = sprokit.pipeline.process_registry.ProcessRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_process('kw_print_number_process', 'A Simple Kwiver Test Process', kw_print_number_process)

    reg.mark_module_as_loaded(module_name)


