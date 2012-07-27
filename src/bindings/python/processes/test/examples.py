#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.pipeline import process


class TestPythonProcess(process.PythonProcess):
    def __init__(self, conf):
        process.PythonProcess.__init__(self, conf)


class PythonPrintNumberProcess(process.PythonProcess):
    def __init__(self, conf):
        process.PythonProcess.__init__(self, conf)

        self.declare_configuration_key(
            'output',
            '',
            'The path of the file to output to.')

        flags = process.PortFlags()

        flags.add(self.flag_required)

        self.declare_input_port(
            'input',
            'integer',
            flags,
            'Where numbers are read from.')

    def _configure(self):
        path = self.config_value('output')

        if not path:
            raise RuntimeError('The path given was empty')

        self.fout = open(path, 'w+')
        self.fout.flush()

        self._base_configure()

    def _reset(self):
        self.fout.close()

        self._base_reset()

    def _step(self):
        num = self.grab_value_from_port('input')

        self.fout.write('%d\n' % num)

        self._base_step()
