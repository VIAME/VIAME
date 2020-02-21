from unittest import TestCase
from kwiver.vital.algo import FeatureDescriptorIO
import nose.tools
import tempfile
from kwiver.vital.modules import modules
import os
from unittest.mock import Mock

class TestVitalFeatureDescriptorIO(TestCase):
    def setUp(self):
        modules.load_known_modules()
        self.instance = FeatureDescriptorIO.create("SimpleFeatureDescriptorIO")
        #TODO: Replace these mocks with the actual vital types
        # The successful tests in this file are not indicators of anything being
        # tested they are failing because the function signature does not match
        # with what pybind11 expects. We would modify these tests once the
        # vital types bindings are complete so that we can test for the desired
        # behavior
        self.feature_set = Mock()
        self.descriptor_set = Mock()

    @nose.tools.raises(Exception)
    def test_load_nonexistant(self):
        self.instance.load("nonexistant_filename.txt", self.feature_set,
                           self.descriptor_set)

    @nose.tools.raises(Exception)
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.load(directory_name, self.feature_set,
                               self.descriptor_set)

    @nose.tools.raises(Exception)
    def test_save_nonexistant(self):
        self.instance.save("nonexistant_filename.txt", self.feature_set,
                           self.descriptor_set)

    @nose.tools.raises(Exception)
    def test_save_directory(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.instance.save(directory_name, self.feature_set,
                               self.descriptor_set)
