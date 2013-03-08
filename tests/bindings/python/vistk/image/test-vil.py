#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.image.vil
    except:
        test_error("Failed to import the vil module")


def test_vil_to_numpy():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (height, width, planes)

    types = [ (test_image.make_image_bool, np.bool)
            , (test_image.make_image_uint8_t, np.uint8)
            , (test_image.make_image_float, np.float32)
            , (test_image.make_image_double, np.double)
            , (test_image.make_image_base, np.uint8)
            ]

    for f, t in types:
        i = vil.vil_to_numpy(f(width, height, planes))

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

    shape = (height, width, planes)

    size = width * height * planes

    types = [ (test_image.take_image_bool, vil.numpy_to_vil_bool, np.bool)
            , (test_image.take_image_uint8_t, vil.numpy_to_vil_uint8_t, np.uint8)
            , (test_image.take_image_float, vil.numpy_to_vil_float, np.float32)
            , (test_image.take_image_double, vil.numpy_to_vil_double, np.double)
            , (test_image.take_image_base, vil.numpy_to_vil, np.uint8)
            ]

    for f, c, t in types:
        i = np.zeros(shape, dtype=t)

        sz = f(c(i))

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

            self.declare_input_port(self.input_port, self.type_any, required, 'image port')

        def _step(self):
            from vistk.image import vil
            from vistk.pipeline import datum
            from vistk.pipeline import edge
            import numpy as np

            dat = self.grab_datum_from_port(self.input_port)
            img = vil.vil_to_numpy(dat.get_datum())

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
    from vistk.pipeline import scheduler_registry
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 1

    shape = (height, width, planes)

    types = [ (test_image.save_image_bool, vil.numpy_to_vil_bool, 'bool', np.bool)
            , (test_image.save_image_uint8_t, vil.numpy_to_vil_uint8_t, 'byte', np.uint8)
            , (test_image.save_image_float, vil.numpy_to_vil_float, 'float', np.float32)
            , (test_image.save_image_double, vil.numpy_to_vil_double, 'double', np.double)
            ]

    modules.load_known_modules()
    reg = process_registry.ProcessRegistry.self()
    sreg = scheduler_registry.SchedulerRegistry.self()

    sched_type = 'sync'

    for f, c, pt, t in types:
        from vistk.pipeline import config
        from vistk.pipeline import pipeline
        from vistk.pipeline import process

        a = np.zeros(shape, dtype=t)

        lname = 'test-python-vil-datum-%s.txt' % pt
        fname = 'test-python-vil-datum-%s.tiff' % pt

        if not f(c(a), fname):
            test_error("Failed to save '%s' image" % pt)
            continue

        with open(lname, 'w+') as f:
            f.write('%s\n' % fname)

        p = pipeline.Pipeline()

        list_name = 'list'
        read_name = 'read'
        verify_name = 'verify'

        c = config.empty_config()

        c['input'] = lname

        proc_type = 'filelist_reader'

        l = reg.create_process(proc_type, list_name, c)

        c = config.empty_config()

        c['pixtype'] = pt
        c['pixfmt'] = 'grayscale'

        proc_type = 'image_reader'

        r = reg.create_process(proc_type, read_name, c)

        c[process.PythonProcess.config_name] = verify_name

        v = create_verify_process(c, shape, t)

        p.add_process(l)
        p.add_process(r)
        p.add_process(v)

        pport = 'path'
        iport = 'image'

        p.connect(list_name, pport,
                  read_name, pport)
        p.connect(read_name, iport,
                  verify_name, iport)

        try:
            p.setup_pipeline()
        except BaseException:
            import sys

            e = sys.exc_info()[1]

            test_error("Could not initialize pipeline: '%s'" % str(e))
            continue

        s = sreg.create_scheduler(sched_type, p)

        try:
            s.start()
            s.wait()
        except BaseException:
            import sys

            e = sys.exc_info()[1]

            test_error("Could not execute pipeline: '%s'" % str(e))

        v.check()


def test_memory():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (height, width, planes)

    types = [ (test_image.make_image_bool, test_image.pass_image_bool, vil.numpy_to_vil_bool, np.bool)
            , (test_image.make_image_uint8_t, test_image.pass_image_uint8_t, vil.numpy_to_vil_uint8_t, np.uint8)
            , (test_image.make_image_float, test_image.pass_image_float, vil.numpy_to_vil_float, np.float32)
            , (test_image.make_image_double, test_image.pass_image_double, vil.numpy_to_vil_double, np.double)
            , (test_image.make_image_base, test_image.pass_image_base, vil.numpy_to_vil, np.uint8)
            ]

    for _, f, c, t in types:
        a = np.zeros(shape, dtype=t)

        x = f(c(a))
        y = f(x)
        z = f(y)
        a = f(z)

    for m, f, _, t in types:
        a = m(width, height, planes)

        x = f(a)
        y = f(x)
        z = f(y)
        a = f(z)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    tests = \
        { 'import': test_import
        , 'vil_to_numpy': test_vil_to_numpy
        , 'numpy_to_vil': test_numpy_to_vil
        , 'datum': test_datum
        , 'memory': test_memory
        }

    from vistk.test.test import *

    run_test(testname, tests)
