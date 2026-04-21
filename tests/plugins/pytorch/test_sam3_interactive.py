"""
Test SAM3 interactive segmentation (point + text query).

Requires SAM3 model pack to be installed.  Run from the install directory:

    cd /path/to/viame/build/install
    source setup_viame.sh
    python -m pytest /path/to/test_sam3_interactive.py -v
"""

import os
import sys
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INSTALL_DIR = os.environ.get(
    "VIAME_INSTALL",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "build", "install"),
)
CONFIGS_DIR = os.path.join(INSTALL_DIR, "configs", "pipelines")
EXAMPLES_DIR = os.path.join(INSTALL_DIR, "examples", "example_imagery", "small_example_image_set1")
TEST_IMAGE = os.path.join(EXAMPLES_DIR, "03142_D20160630-T162918.657_00-0C-DF-06-40-BF.png")
SAM3_SEGMENTER_CONF = os.path.join(CONFIGS_DIR, "interactive_segmenter_sam3.conf")
SAM3_TEXT_QUERY_CONF = os.path.join(CONFIGS_DIR, "interactive_text_query_sam3.conf")
SAM3_WEIGHTS = os.path.join(CONFIGS_DIR, "models", "sam3_weights.pt")


_plugins_loaded = False


def _ensure_plugins():
    global _plugins_loaded
    if _plugins_loaded:
        return
    from kwiver.vital import modules as vital_modules
    vital_modules.load_known_modules()
    _plugins_loaded = True


def _skip_if_no_model():
    if not os.path.exists(SAM3_WEIGHTS):
        pytest.skip("SAM3 model pack not installed")


def _skip_if_no_image():
    if not os.path.exists(TEST_IMAGE):
        pytest.skip("Example imagery not found")


def _load_image():
    """Load test image via KWIVER."""
    from kwiver.vital.algo import ImageIO
    reader = ImageIO.create("vxl")
    reader.set_configuration(reader.get_configuration())
    return reader.load(TEST_IMAGE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSAM3Segmenter:
    """Test SAM3 point-based segmentation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_plugins()
        _skip_if_no_model()
        _skip_if_no_image()

    def test_segment_with_point(self):
        """A single positive point should produce a valid mask polygon."""
        from kwiver.vital.algo import SegmentViaPoints
        from kwiver.vital.types import Point2d
        from kwiver.vital.config import config as vital_config

        cfg = vital_config.read_config_file(SAM3_SEGMENTER_CONF)

        # Resolve relative model paths
        from pathlib import Path
        config_dir = Path(SAM3_SEGMENTER_CONF).parent
        for key in cfg.available_values():
            for pk in ['checkpoint', 'model_config']:
                if pk in key:
                    val = cfg.get_value(key)
                    if val and not os.path.isabs(val):
                        resolved = config_dir / val
                        if resolved.exists():
                            cfg.set_value(key, str(resolved))

        impl = cfg.get_value("segment_via_points:type")
        algo = SegmentViaPoints.create(impl)
        algo.set_configuration(cfg.subblock("segment_via_points:" + impl))

        image = _load_image()

        # Click roughly in the center of a known fish (~310, 290)
        points = [Point2d(310.0, 290.0)]
        labels = [1]  # positive

        result = algo.segment(image, points, labels)

        assert result is not None
        # Should return at least one detection
        assert len(result) > 0
        det = list(result)[0]
        bb = det.bounding_box
        assert bb.width() > 5
        assert bb.height() > 5
        print(f"  Segmentation bbox: ({bb.min_x():.0f},{bb.min_y():.0f})-({bb.max_x():.0f},{bb.max_y():.0f})")


class TestSAM3TextQuery:
    """Test SAM3 text-based detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_plugins()
        _skip_if_no_model()
        _skip_if_no_image()

    def test_text_query_fish(self):
        """Text query 'fish' should find detections in the fish image."""
        from kwiver.vital.algo import PerformTextQuery
        from kwiver.vital.config import config as vital_config

        cfg = vital_config.read_config_file(SAM3_TEXT_QUERY_CONF)

        from pathlib import Path
        config_dir = Path(SAM3_TEXT_QUERY_CONF).parent
        for key in cfg.available_values():
            for pk in ['checkpoint', 'model_config', 'grounding_model_id']:
                if pk in key:
                    val = cfg.get_value(key)
                    if val and not os.path.isabs(val):
                        resolved = config_dir / val
                        if resolved.exists():
                            cfg.set_value(key, str(resolved))

        impl = cfg.get_value("perform_text_query:type")
        algo = PerformTextQuery.create(impl)
        algo.set_configuration(cfg.subblock("perform_text_query:" + impl))

        image = _load_image()

        track_sets = algo.perform_query("fish", [image])

        assert track_sets is not None
        assert len(track_sets) == 1

        ts = track_sets[0]
        tracks = ts.tracks()
        assert len(tracks) > 0, "Expected at least one fish detection"

        for track in tracks:
            for state in track:
                det = state.detection()
                bb = det.bounding_box
                assert bb.width() > 5
                assert bb.height() > 5
                print(f"  Track {track.id}: bbox=({bb.min_x():.0f},{bb.min_y():.0f})-"
                      f"({bb.max_x():.0f},{bb.max_y():.0f}) conf={det.confidence:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
