#ckwg +5
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.pipeline import process


class TestPythonProcess(process.PythonProcess):
    def __init__(self, conf):
        process.PythonProcess.__init__(self, conf)
