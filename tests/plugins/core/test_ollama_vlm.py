"""The whole-sequence VLM text query: detections for the tracker, and the
refiner that keeps or drops existing tracks without reusing a track ID.
Ollama is stubbed out."""
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("kwiver.vital.algo")
from kwiver.vital.types import (  # noqa: E402
    BoundingBoxD, DetectedObject, Image, ImageContainer, ObjectTrackSet,
    ObjectTrackState, Timestamp, Track,
)
from viame.core import ollama_vlm  # noqa: E402


def _refiner(replace):
    refiner = ollama_vlm.OllamaVlmRefiner()
    cfg = refiner.get_configuration()
    cfg.set_value("text_query", "fish")
    cfg.set_value("replace_existing", str(replace))
    refiner.set_configuration(cfg)
    return refiner


def _frame(frame):
    ts = Timestamp()
    ts.set_frame(frame)
    ts.set_time_usec(frame * 1000)
    return ts, ImageContainer(Image(np.zeros((100, 200, 3), dtype=np.uint8)))


def _track(tid, frame):
    track = Track(id=tid)
    det = DetectedObject(BoundingBoxD(0, 0, 5, 5), 1.0)
    track.append(ObjectTrackState(frame, frame * 1000, det))
    return track


FOUND = [([10.0, 20.0, 30.0, 40.0], "cod"), ([50.0, 50.0, 60.0, 60.0], None)]


def test_replace_drops_existing_and_labels_new():
    refiner = _refiner(True)
    with patch.object(ollama_vlm, "detect_objects", return_value=FOUND):
        out = refiner.refine(*_frame(0), ObjectTrackSet([_track(7, 0)]))
    tracks = out.tracks()
    assert [t.id for t in tracks] == [1, 2]
    labels = [t[0].detection().type.get_most_likely_class() for t in tracks]
    assert labels == ["cod", "fish"]


def test_keep_existing_remaps_late_colliding_input_id():
    refiner = _refiner(False)
    with patch.object(ollama_vlm, "detect_objects", return_value=FOUND[:1]):
        first = refiner.refine(*_frame(0), ObjectTrackSet([]))
        second = refiner.refine(*_frame(1), ObjectTrackSet([_track(1, 1)]))
    assert [t.id for t in first.tracks()] == [1]
    ids = [t.id for t in second.tracks()]
    assert len(ids) == len(set(ids)) == 2
    assert 1 not in ids


def test_detector_returns_labelled_detections():
    detector = ollama_vlm.OllamaVlmDetector()
    cfg = detector.get_configuration()
    cfg.set_value("text_query", "fish")
    cfg.set_value("max_new_objects", "1")
    detector.set_configuration(cfg)
    with patch.object(ollama_vlm, "detect_objects", return_value=FOUND):
        dets = list(detector.detect(_frame(0)[1]))
    assert len(dets) == 1
    assert dets[0].type.get_most_likely_class() == "cod"
    assert dets[0].bounding_box.max_x() == 30.0
