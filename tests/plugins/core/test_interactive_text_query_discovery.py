"""A text-query add-on installed after the service started is found on the
first text query, without a restart."""
from pathlib import Path
from unittest.mock import Mock

import pytest

kwiver = pytest.importorskip("kwiver.vital.algo")
from viame.core import interactive_segmentation as seg  # noqa: E402


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "interactive_segmenter_default.conf").write_text(
        "block segment_via_points\n  type = grabcut\nendblock\n"
    )
    return tmp_path


def test_no_text_query_config_gives_none(config_dir):
    conf = config_dir / "interactive_segmenter_default.conf"
    assert seg.load_text_query_algo_from_config([str(conf)]) is None


def test_sibling_installed_later_is_found(config_dir):
    conf = config_dir / "interactive_segmenter_default.conf"
    try:
        kwiver.PerformTextQuery.create("sam3")
    except Exception:
        pytest.skip("sam3 text query plugin not built")
    (config_dir / "interactive_text_query_default.conf").write_text(
        "block perform_text_query\n  type = sam3\n  sam3:device = cpu\nendblock\n"
    )
    algo = seg.load_text_query_algo_from_config([str(conf)])
    assert algo is not None


def test_service_accepts_text_query_once_algo_is_attached():
    service = seg.InteractiveSegmentationService(segment_via_points_algo=Mock())
    assert not service.has_text_query()
    with pytest.raises(ValueError, match="requires a perform_text_query"):
        service.handle_request({"command": "text_query"})
    service.set_text_query_algo(Mock())
    assert service.has_text_query()
    with pytest.raises(ValueError, match="image_path is required"):
        service.handle_request({"command": "text_query", "text": "fish"})
