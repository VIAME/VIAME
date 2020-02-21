from unittest import TestCase
from kwiver.vital.algo import ImageIO
import nose.tools
import tempfile
from kwiver.vital.modules import modules
import os
from kwiver.vital.types import ImageContainer
import  numpy as np

class TestVitalImageIO(TestCase):
    def setUp(self):
        modules.load_known_modules()
        self.instance = ImageIO.create("SimpleImageIO")

    @nose.tools.raises(RuntimeError)
    def test_load_nonexistant(self):
        self.instance.load("nonexistant_filename.txt")

    @nose.tools.raises(RuntimeError)
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load(directory_name)

    @nose.tools.raises(RuntimeError)
    def test_load_metadata_nonexistant(self):
        self.instance.load_metadata("nonexistant_filename.txt")

    @nose.tools.raises(RuntimeError)
    def test_load_metadata_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load_metadata(directory_name)

    @nose.tools.raises(RuntimeError)
    def test_save_nonexistant(self):
        dummy_image = np.zeros([100, 100])
        image_container = ImageContainer.fromarray(dummy_image)
        self.instance.save("nonexistant_filename.txt", image_container)

    @nose.tools.raises(RuntimeError)
    def test_save_directory(self):
        dummy_image = np.zeros([100, 100])
        image_container = ImageContainer.fromarray(dummy_image)
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.save(directory_name, image_container)
