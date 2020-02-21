from unittest import TestCase
from kwiver.vital.algo import Transform2DIO
import nose.tools
import tempfile
from kwiver.vital.modules import modules
from unittest.mock import Mock

class TestVitalTransform2DIO(TestCase):
    def setUp(self):
        modules.load_known_modules()
        self.instance = Transform2DIO.create("SimpleTransform2DIO")
        #TODO: Replace these mocks with the actual vital types
        # The successful tests in this file are not indicators of anything being
        # tested they are failing because the function signature does not match
        # with what pybind11 expects. We would modify these tests once the
        # vital types bindings are complete so that we can test for the desired
        # behavior
        self.transform_2d = Mock()

    @nose.tools.raises(RuntimeError)
    def test_load_nonexistant(self):
        self.instance.load("nonexistant_filename.txt")

    @nose.tools.raises(RuntimeError)
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load(directory_name)

    @nose.tools.raises(Exception)
    def test_save_nonexistant(self):
        self.instance.save("nonexistant_filename.txt",
                           self.transform_2d)

    @nose.tools.raises(Exception)
    def test_save_directory(self):
        # Create a mock transform2d object
        # save() requires a second argument
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.save(directory_name, self.transform_2d)
