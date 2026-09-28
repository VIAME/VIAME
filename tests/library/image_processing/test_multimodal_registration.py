"""Multimodal process registration and thermal normalization contracts."""
import numpy as np

from viame.image_processing import multimodal_registration as registration
from viame import config


def test_process_ports_construct():
    proc = registration.register_frames_process(config.empty_config())
    assert 'optical_to_thermal_homog' in proc.output_ports()
    assert 'thermal_to_optical_homog' in proc.output_ports()


def test_normalization_does_not_wrap_bright_pixels_or_divide_by_zero():
    values = np.arange(1000, dtype=np.uint16).reshape(20, 50)
    result = registration.normalize_thermal(values)
    assert result.min() == 0
    assert result.max() == 255
    assert result[-1, -1] == 255
    assert np.all(np.diff(result.ravel().astype(int)) >= 0)
    assert np.all(registration.normalize_thermal(np.ones((3, 4))) == 0)
    assert registration.normalize_thermal(None) is None


def test_failed_registration_emits_every_output(monkeypatch):
    from viame.types import ImageContainer
    outputs = {}

    class Step:
        _good_match_percent = 0.5
        _ratio_test = 0.8
        _match_height = 100
        _min_matches = 4
        _min_inliers = 4

        def grab_input_using_trait(self, name):
            return ImageContainer.fromarray(np.ones((8, 8, 3), dtype=np.uint8))

        def push_to_port_using_trait(self, name, value):
            outputs[name] = value

        def push_datum_to_port(self, name, value):
            outputs[name] = value

        def _base_step(self):
            pass

    monkeypatch.setattr(registration, 'compute_transform', lambda *a, **k: (False, None, None))
    registration.register_frames_process._step(Step())
    assert set(outputs) == {'warped_optical_image', 'warped_thermal_image',
                            'optical_to_thermal_homog', 'thermal_to_optical_homog'}
    assert outputs['thermal_to_optical_homog'].type() == registration.datum.DatumType.empty
