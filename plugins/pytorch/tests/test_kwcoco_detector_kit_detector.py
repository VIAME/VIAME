"""
Smoke tests for kwcoco_detector_kit_detector.py — the VIAME plugin that wraps
the kit's ONNX predictor.

Requires
--------
- kwiver (``from kwiver.vital.algo import ImageObjectDetector``)
- viame (``from viame.pytorch.utilities import ...``)
- kwcoco_detector_kit installed in this Python environment
- A real ONNX export for the integration-level tests

Run from the VIAME checkout root::

    pytest plugins/pytorch/tests/test_kwcoco_detector_kit_detector.py -v

Or from anywhere with::

    pytest /path/to/VIAME/plugins/pytorch/tests/test_kwcoco_detector_kit_detector.py -v

Skip the GPU tests with::

    pytest ... -m "not cuda"
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the pytorch plugin directory importable when running pytest directly
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

# ---------------------------------------------------------------------------
# Module-level guards — defined before pytestmark uses them
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


pytestmark = [
    pytest.mark.skipif(
        not _can_import("kwiver.vital.algo"),
        reason="kwiver not installed",
    ),
    pytest.mark.skipif(
        not _can_import("viame.pytorch.utilities"),
        reason="viame Python package not installed",
    ),
    pytest.mark.skipif(
        not _can_import("kwcoco_detector_kit"),
        reason="kwcoco_detector_kit not installed",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KCD_TRAINING_ROOT = Path(os.environ.get("KCD_TRAINING_ROOT",
                                         "/data/users/jon.crall/kcd_sealion"))


def _find_real_onnx_package() -> Path | None:
    env = os.environ.get("KCD_TEST_ONNX_PACKAGE")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    if not _KCD_TRAINING_ROOT.exists():
        return None
    for candidate in sorted(_KCD_TRAINING_ROOT.rglob("*.onnx")):
        if candidate.stat().st_size > 1_000_000:
            return candidate.parent
    return None


def _make_kwiver_image(h=64, w=64):
    """Synthetic kwiver ImageContainer (RGB uint8)."""
    from kwiver.vital.types import Image, ImageContainer
    arr = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
    return ImageContainer(Image(arr))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector_class():
    from kwcoco_detector_kit_detector import KwcocoDetectorKitDetector
    return KwcocoDetectorKitDetector


@pytest.fixture(scope="module")
def real_onnx_package() -> Path:
    pkg = _find_real_onnx_package()
    if pkg is None:
        pytest.skip(
            "No real ONNX export found. "
            "Set KCD_TEST_ONNX_PACKAGE or run: "
            "python -m kwcoco_detector_kit export-onnx <workdir>"
        )
    return pkg


# ---------------------------------------------------------------------------
# Tests: plugin import and structure
# ---------------------------------------------------------------------------

def test_plugin_imports():
    """The plugin module can be imported without errors."""
    import kwcoco_detector_kit_detector  # noqa: F401


def test_plugin_exports_vital_register():
    """__vital_algorithm_register__ is present and callable."""
    from kwcoco_detector_kit_detector import __vital_algorithm_register__
    assert callable(__vital_algorithm_register__)


def test_detector_inherits_image_object_detector(detector_class):
    from kwiver.vital.algo import ImageObjectDetector
    assert issubclass(detector_class, ImageObjectDetector)


def test_detector_instantiates(detector_class):
    obj = detector_class()
    assert obj is not None


def test_detector_get_configuration_returns_keys(detector_class):
    """get_configuration() exposes all expected config keys."""
    obj = detector_class()
    cfg = obj.get_configuration()
    for key in ("package", "device", "score_thresh", "nms_thresh"):
        assert cfg.has_value(key), f"config key {key!r} missing"


def test_check_configuration_rejects_empty(detector_class):
    """check_configuration returns False when package is unset."""
    obj = detector_class()
    cfg = obj.get_configuration()
    assert not obj.check_configuration(cfg)


def test_check_configuration_accepts_package(detector_class, tmp_path):
    """check_configuration returns True when package is non-empty string."""
    obj = detector_class()
    cfg = obj.get_configuration()
    cfg.set_value("package", str(tmp_path))
    assert obj.check_configuration(cfg)


# ---------------------------------------------------------------------------
# Tests: set_configuration + detect on real ONNX
# ---------------------------------------------------------------------------

@pytest.mark.requires_onnx
def test_set_configuration_builds_predictor(detector_class, real_onnx_package):
    """set_configuration loads the ONNX model without error."""
    pytest.importorskip("onnxruntime")
    obj = detector_class()
    result = obj.set_configuration(dict(
        package=str(real_onnx_package),
        device="cpu",
        score_thresh="0.30",
    ))
    assert result is True
    assert obj._predictor is not None


@pytest.mark.requires_onnx
def test_detect_synthetic_image(detector_class, real_onnx_package):
    """detect() runs on a synthetic kwiver image and returns a DetectedObjectSet."""
    pytest.importorskip("onnxruntime")
    from kwiver.vital.types import DetectedObjectSet

    obj = detector_class()
    obj.set_configuration(dict(
        package=str(real_onnx_package),
        device="cpu",
        score_thresh="0.10",
    ))

    image_data = _make_kwiver_image(640, 640)
    result = obj.detect(image_data)

    assert isinstance(result, DetectedObjectSet)
    print(f"  detect() returned {len(result)} detections on synthetic 640x640 image")


@pytest.mark.requires_onnx
def test_detect_returns_valid_boxes(detector_class, real_onnx_package):
    """All returned detections have valid bounding boxes and scores in range."""
    pytest.importorskip("onnxruntime")
    obj = detector_class()
    obj.set_configuration(dict(
        package=str(real_onnx_package),
        device="cpu",
        score_thresh="0.05",
    ))

    image_data = _make_kwiver_image(640, 640)
    result = obj.detect(image_data)

    for det in result:
        bb = det.bounding_box
        assert bb.width() > 0, f"zero-width box: {bb}"
        assert bb.height() > 0, f"zero-height box: {bb}"
        assert 0.0 <= det.confidence <= 1.0, f"score out of range: {det.confidence}"


@pytest.mark.cuda
@pytest.mark.requires_onnx
def test_detect_cuda_runs(detector_class, real_onnx_package):
    """detect() runs with device=cuda when a GPU is available."""
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider not available")

    obj = detector_class()
    obj.set_configuration(dict(
        package=str(real_onnx_package),
        device="cuda",
        score_thresh="0.30",
    ))

    image_data = _make_kwiver_image(640, 640)
    result = obj.detect(image_data)
    print(f"  CUDA detect() returned {len(result)} detections")
