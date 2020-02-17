from unittest import TestCase
from kwiver.vital.algo import Transform2DIO
import nose.tools
import tempfile
from kwiver.vital.modules import modules


class TestVitalTransform2DIO(TestCase):
    def setUp(self):
        modules.load_known_modules()
        self.instance = Transform2DIO.create("SimpleTransform2DIO")

    @nose.tools.raises(Exception)
    def test_load_nonexistant(self):
        self.instance.load("nonexistant_filename.txt")

    @nose.tools.raises(Exception)
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load(directory_name)

    @nose.tools.raises(Exception)
    def test_save_nonexistant(self):
        self.instance.save("nonexistant_filename.txt")

    @nose.tools.raises(Exception)
    def test_save_directory(self):
        # Create a mock transform2d object
        # save() requires a second argument
        transform_2d_mock = Mock()
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.save(directory_name, transform_2d_mock)
