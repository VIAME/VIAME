# gkwg +5
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
#

# from sprokit.utilities import homography
#from sprokit.utilities import timestamp
from __future__ import print_function
import kwiver
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

# from libkwiver_python_convert_homography.homograpy import HomograpyTra
import os.path
# import libkwiver_python_convert_homography

class HomographyWriterProcess(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # declare our configuration items
        self.declare_configuration_key(
            'output',
            'output-homog.txt',
            'The output file name.')


        required = kwiver.sprokit.pipeline.process.PortFlags()
        required.add(self.flag_required)

        # create input ports
        self.add_port_trait('homography', 'homography_src_to_ref', 'Input homographies')
        self.declare_input_port_using_trait('homography', required)

    # ----------------------------------------------------------------
    def _configure(self):
        path = self.config_value('output')

        self.fout = open(path, 'w+')

        self._base_configure()

    # ----------------------------------------------------------------
    def _step(self):
        h = self.grab_input_using_trait('homography')

        for r in [ 0, 1, 2 ]:
            for c in [ 0, 1, 2 ]:
                val = h.get( r, c )
                print(val, end=' ')
                self.fout.write( '%.20g ' % val )

        print(h.from_id, h.to_id)
        self.fout.write( '%d %d\n' % (h.from_id, h.to_id) )
        self.fout.flush()
        ## t = h # .transform()

       # values = tuple([t.get(i // 3, i % 3) for i in range(9)])
       # print("XXXXXXX homography", values, "\n")

        #self.fout.write('\n%.20g %.20g %.20g\n%.20g %.20g %.20g\n%.20g %.20g %.20g\n\n' % values)

        self._base_step()


# ----------------------------------------------------------------
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.write_homography'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('kw_write_homography',
                                'A Simple Kwiver homography writer',
                                HomographyWriterProcess)

    process_factory.mark_process_module_as_loaded(module_name)
