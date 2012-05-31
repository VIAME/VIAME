#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.image.vil
    except:
        test_error("Failed to import the vil module")


def test_create():
    from vistk.image import vil

    try:
        config.empty_config()
    except:
        test_error("Failed to create an empty configuration")

    config.ConfigKey()
    config.ConfigKeys()
    config.ConfigValue()


def test_vil_to_numpy():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    types = [ (test_image.make_image_bool, np.bool)
            , (test_image.make_image_uint8_t, np.uint8)
            , (test_image.make_image_float, np.float32)
            , (test_image.make_image_double, np.double)
            ]

    for f, t in types:
        i = f(width, height, planes)

        if not i.dtype == t:
            test_error("Wrong type returned: got: '%s' expected: '%s'" % (i.dtype, t))

        if not i.ndim == 3:
            test_error("Did not get a 3-dimensional array: got: '%d' expected: '%d'" % (i.dim, 3))

        if not i.shape == shape:
            test_error("Did not get expected array sizes: got '%s' expected: '%s'" % (str(i.shape), str(shape)))


def test_numpy_to_vil():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    size = width * height * planes

    types = [ (test_image.take_image_bool, np.bool)
            , (test_image.take_image_uint8_t, np.uint8)
            , (test_image.take_image_float, np.float32)
            , (test_image.take_image_double, np.double)
            ]

    for f, t in types:
        i = np.zeros(shape, dtype=t)

        sz = f(i)

        if not sz == size:
            test_error("Wrong size calculated: got: '%d' expected: '%d'" % (sz, size))


def create_verify_process(c, shape, dtype):
    from vistk.pipeline import process

    class VerifyProcess(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

            self.got_image = False
            self.same_image_type = False
            self.same_image_size = False

            self.input_port = 'image'

            required = process.PortFlags()
            required.add(self.flag_required)

            info = process.PortInfo(self.type_any, required, 'image port')

            self.declare_input_port(self.input_port, info)

        def _step(self):
            from vistk.pipeline import datum
            from vistk.pipeline import edge
            import numpy as np

            dat = self.grab_datum_from_port(self.input_port)
            img = dat.get_datum()

            if isinstance(img, np.ndarray):
                self.got_image = True
                if dtype == img.dtype:
                    self.same_image_type = True
                if shape == img.shape:
                    self.same_image_size = True

            self._base_step()

        def check(self):
            if not self.got_image:
                test_error("Could not grab an image from the datum")
            if not self.same_image_type:
                test_error("Input image was not of the expected type")
            if not self.same_image_size:
                test_error("Input image was not of the expected size")

    return VerifyProcess(c)


def test_datum():
    from vistk.image import vil
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry
    from vistk.pipeline import schedule_registry
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 1

    shape = (width, height, planes)

    types = [ (test_image.save_image_bool, 'bool', np.bool)
            , (test_image.save_image_uint8_t, 'byte', np.uint8)
            , (test_image.save_image_float, 'float', np.float32)
            , (test_image.save_image_double, 'double', np.double)
            ]

    modules.load_known_modules()
    reg = process_registry.ProcessRegistry.self()
    sreg = schedule_registry.ScheduleRegistry.self()

    sched_type = 'sync'

    for f, pt, t in types:
        from vistk.pipeline import config
        from vistk.pipeline import pipeline
        from vistk.pipeline import process

        a = np.zeros(shape, dtype=t)

        lname = 'test-python-vil-datum-%s.txt' % pt
        fname = 'test-python-vil-datum-%s.tiff' % pt

        if not f(a, fname):
            test_error("Failed to save '%s' image" % pt)
            continue

        with open(lname, 'w+') as f:
            f.write('%s\n' % fname)

        p = pipeline.Pipeline()

        read_name = 'read'
        verify_name = 'verify'

        c = config.empty_config()

        c['input'] = lname
        c['pixtype'] = pt
        c['pixfmt'] = 'grayscale'
        c['verify'] = 'true'
        c[process.PythonProcess.config_name] = read_name

        proc_type = 'image_reader'

        r = reg.create_process(proc_type, c)

        c[process.PythonProcess.config_name] = verify_name

        v = create_verify_process(c, shape, t)

        p.add_process(r)
        p.add_process(v)

        port = 'image'

        p.connect(read_name, port,
                  verify_name, port)

        try:
            p.setup_pipeline()
        except BaseException as e:
            test_error("Could not initialize pipeline: '%s'" % str(e))
            continue

        s = sreg.create_schedule(sched_type, p)

        try:
            s.start()
            s.wait()
        except BaseException as e:
            test_error("Could not execute pipeline: '%s'" % str(e))

        v.check()


def test_memory():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    types = [ (test_image.make_image_bool, test_image.pass_image_bool, np.bool)
            , (test_image.make_image_uint8_t, test_image.pass_image_uint8_t, np.uint8)
            , (test_image.make_image_float, test_image.pass_image_float, np.float32)
            , (test_image.make_image_double, test_image.pass_image_double, np.double)
            ]

    for _, f, t in types:
        a = np.zeros(shape, dtype=t)

        x = f(a)
        y = f(x)
        z = f(y)
        a = f(z)

    for m, f, t in types:
        a = m(width, height, planes)

        x = f(a)
        y = f(x)
        z = f(y)
        a = f(z)


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'vil_to_numpy':
        test_vil_to_numpy()
    elif testname == 'numpy_to_vil':
        test_numpy_to_vil()
    elif testname == 'datum':
        test_datum()
    elif testname == 'memory':
        test_memory()
    else:
        test_error("No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from vistk.test.test import *

    try:
        main(testname)
    except BaseException as e:
        test_error("Unexpected exception: %s" % str(e))
