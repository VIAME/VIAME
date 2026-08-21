# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for the COCO detection and track writers.

These pin the single COCO profile VIAME emits, which has to stay readable by
plain MS-COCO tooling, by kwcoco, and by DIVE's importer at the same time:

- every table VIAME allocates ids for is 1-based and contiguous
- every image carries a ``file_name`` and a unique ``name``
- annotations carry ``area`` and ``iscrowd`` alongside ``bbox``
- segmentations are image-coordinate polygons, never bounding-box RLE
- images are ordered by time, not by the order tracks happen to be visited

kwcoco is deliberately not imported here; the point is that the writer output
conforms on its own.
"""

import json
import os

import numpy as np
import pytest

from viame.core import utilities_coco as uc


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _assert_one_based_contiguous(rows, label):
    ids = sorted(row["id"] for row in rows)
    assert ids == list(range(1, len(rows) + 1)), (
        "{} ids must be 1-based and contiguous, got {}".format(label, ids))


def _assert_profile(doc, expect_video=False):
    """Assert the invariants every VIAME COCO document must satisfy."""
    for key in ("info", "licenses", "images", "annotations", "categories"):
        assert key in doc, "missing top-level '{}'".format(key)

    _assert_one_based_contiguous(doc["images"], "image")
    _assert_one_based_contiguous(doc["annotations"], "annotation")
    _assert_one_based_contiguous(doc["categories"], "category")
    if expect_video:
        _assert_one_based_contiguous(doc["videos"], "video")

    names = [image["name"] for image in doc["images"]]
    assert len(names) == len(set(names)), "image names must be unique"
    for image in doc["images"]:
        assert image["file_name"], "every image needs a file_name"

    images_by_id = {image["id"]: image for image in doc["images"]}
    for ann in doc["annotations"]:
        # Exactly the dereference DIVE's importer performs.
        image = images_by_id[ann["image_id"]]
        assert "file_name" in image and "frame_index" in image
        assert isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
        assert "area" in ann and "iscrowd" in ann
        assert "category_id" in ann
        # A dict segmentation is RLE, which DIVE cannot decode.
        assert not isinstance(ann.get("segmentation"), dict)


# ----------------------------------------------------------------------
# Geometry helpers (no kwiver required)
# ----------------------------------------------------------------------

def test_mask_to_polygons_returns_image_coordinates():
    """kwiver masks are box-relative; the polygon must land in image space."""
    if not uc._HAS_CV2:
        pytest.skip("OpenCV not available")
    mask = np.zeros((10, 10), np.uint8)
    mask[2:8, 3:9] = 1

    polygons = uc.mask_to_polygons(mask, offset_x=100.0, offset_y=200.0)

    assert len(polygons) == 1
    xs, ys = polygons[0][0::2], polygons[0][1::2]
    assert min(xs) == 103.0 and max(xs) == 108.0
    assert min(ys) == 202.0 and max(ys) == 207.0


def test_polygon_area_and_fallback():
    square = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    assert uc.polygon_area(square) == pytest.approx(100.0)
    assert uc.polygon_area([0.0, 0.0, 1.0, 1.0]) == 0.0
    # No usable polygon falls back to the box area.
    assert uc.segmentation_area(None, 4, 5) == pytest.approx(20.0)
    assert uc.segmentation_area([square], 4, 5) == pytest.approx(100.0)


def test_build_image_list_fills_required_fields():
    entries = uc.build_image_list(
        ["a/img.png", "b/img.png", ""], [""], [""])

    assert [entry["id"] for entry in entries] == [1, 2, 3]
    # Colliding basenames disambiguate rather than repeating a name.
    assert entries[0]["name"] != entries[1]["name"]
    # A frame with no imagery still gets a file_name.
    assert entries[2]["file_name"]


# ----------------------------------------------------------------------
# Writers (require the kwiver bindings)
# ----------------------------------------------------------------------

try:
    from kwiver.vital import types as vital_types
except ImportError:  # the geometry tests above still run without the bindings
    vital_types = None

requires_kwiver = pytest.mark.skipif(
    vital_types is None, reason="kwiver python bindings not available")


def _multiclass_detection(x, y, w, h, pairs):
    """A detection the way a VIAME detector scores one: every class kept."""
    dot = vital_types.DetectedObjectType(
        [name for name, _ in pairs], [score for _, score in pairs])
    return vital_types.DetectedObject(
        vital_types.BoundingBoxD(x, y, x + w, y + h), pairs[0][1], dot)


def _detection(x, y, w, h, label, polygon=None):
    dot = vital_types.DetectedObjectType(label, 1.0)
    det = vital_types.DetectedObject(
        vital_types.BoundingBoxD(x, y, x + w, y + h), 0.9, dot)
    if polygon is not None:
        det.set_flattened_polygon(polygon)
    return det


@pytest.fixture(autouse=True)
def _reset_global_categories():
    """Category ids are shared process-wide between the two writers."""
    uc.global_categories.clear()
    yield
    uc.global_categories.clear()


@requires_kwiver
def test_track_writer_orders_images_by_frame(tmp_path):
    """Images follow time even when the later track is stored first."""
    from viame.core.write_object_track_set_coco import WriteObjectTrackSetCoco

    late = vital_types.Track(id=10)
    for frame in (30, 31):
        late.append(vital_types.ObjectTrackState(
            frame, frame * 1000000, _detection(10, 10, 20, 20, "fish")))

    early = vital_types.Track(id=20)
    for frame in (5, 6):
        early.append(vital_types.ObjectTrackState(
            frame, frame * 1000000,
            _detection(50, 50, 10, 10, "crab",
                       polygon=[50, 50, 60, 50, 60, 60, 50, 60])))

    writer = WriteObjectTrackSetCoco()
    writer.set_configuration(writer.get_configuration())
    writer.video_name = "my_video"

    out = str(tmp_path / "tracks.json")
    writer.open(out)
    # Frame 7 carries no tracks; it should still appear in the images table.
    for frame, name, tracks in [(5, "f005.png", [early]), (6, "f006.png", [early]),
                                (7, "f007.png", []),
                                (30, "f030.png", [late]), (31, "f031.png", [late])]:
        stamp = vital_types.Timestamp()
        stamp.set_frame(frame)
        stamp.set_time_seconds(1000.0 + frame)
        writer.write_set(vital_types.ObjectTrackSet(tracks), stamp, name)
    writer.close()

    doc = _load(out)
    _assert_profile(doc, expect_video=True)

    frame_indices = [image["frame_index"] for image in doc["images"]]
    assert frame_indices == [5, 6, 7, 30, 31]
    assert [image["id"] for image in doc["images"]] == [1, 2, 3, 4, 5]
    assert doc["videos"] == [{"id": 1, "name": "my_video"}]
    assert sorted(track["id"] for track in doc["tracks"]) == [10, 20]

    # track_id keeps the pipeline's own numbering so it matches the CSV writer.
    by_track = {}
    for ann in doc["annotations"]:
        by_track.setdefault(ann["track_id"], []).append(ann["image_id"])
    assert by_track[20] == [1, 2]
    assert by_track[10] == [4, 5]


@requires_kwiver
def test_detection_writer_profile(tmp_path):
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())

    out = str(tmp_path / "detections.json")
    writer.open(out)
    for name, det in [("a/img.png", _detection(1, 2, 3, 4, "fish")),
                      ("b/img.png", _detection(5, 6, 7, 8, "crab"))]:
        writer.write_set(vital_types.DetectedObjectSet([det]), name)
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    assert [(a["id"], a["image_id"]) for a in doc["annotations"]] == [(1, 1), (2, 2)]
    assert doc["annotations"][0]["area"] == pytest.approx(12.0)


def _write_detections(tmp_path, name, frames, video_name=""):
    """Run the detection writer over *frames*, a list of (file_name, dets)."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    writer.video_name = video_name

    out = str(tmp_path / name)
    writer.open(out)
    for file_name, dets in frames:
        writer.write_set(vital_types.DetectedObjectSet(dets), file_name)
    writer.complete()
    writer.close()
    return _load(out)


@requires_kwiver
def test_detection_writer_video_holds_blank_frames(tmp_path):
    """Video names no frame, so blank frames have to hold their slot.

    Dropping them would slide every later detection towards the start of the
    clip; frame_index has to keep matching the frame column viame_csv writes.
    """
    frames = [
        ("", []),
        ("", [_detection(1, 2, 3, 4, "fish")]),
        ("", []),
        ("", []),
        ("", [_detection(5, 6, 7, 8, "fish")]),
        ("", []),
    ]
    doc = _write_detections(tmp_path, "my_clip.json", frames)

    _assert_profile(doc, expect_video=True)
    # The trailing blank is dropped; the interior ones are not.
    assert [image["frame_index"] for image in doc["images"]] == [0, 1, 2, 3, 4]

    images_by_id = {image["id"]: image for image in doc["images"]}
    annotated = sorted(images_by_id[ann["image_id"]]["frame_index"]
                       for ann in doc["annotations"])
    assert annotated == [1, 4]

    # Frames of a video are one video, named for the file when nothing else says.
    assert doc["videos"] == [{"id": 1, "name": "my_clip"}]
    assert all(image["video_id"] == 1 for image in doc["images"])
    assert all(image["file_name"].endswith(".png") for image in doc["images"])


@requires_kwiver
def test_detection_writer_image_list_has_no_video(tmp_path):
    """An image list is not a video: DIVE checks sort order only without one."""
    frames = [("f000.png", []),
              ("f001.png", [_detection(1, 2, 3, 4, "fish")]),
              ("f002.png", [])]
    doc = _write_detections(tmp_path, "detections.json", frames)

    _assert_profile(doc)
    assert "videos" not in doc
    assert not any("video_id" in image for image in doc["images"])
    assert [image["file_name"] for image in doc["images"]] == [
        "f000.png", "f001.png", "f002.png"]


@requires_kwiver
def test_every_scored_class_survives(tmp_path):
    """COCO has one category_id; a VIAME detector scores every class."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    out = str(tmp_path / "detections.json")
    writer.open(out)
    writer.write_set(vital_types.DetectedObjectSet([
        _multiclass_detection(1, 2, 3, 4,
                              [("fish", 0.7), ("crab", 0.2), ("rock", 0.1)])]),
        "f000.png")
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    ann = doc["annotations"][0]

    # The winner still occupies category_id, so plain COCO readers are unaffected.
    names = {category["id"]: category["name"] for category in doc["categories"]}
    assert names[ann["category_id"]] == "fish"

    assert ann["confidence_pairs"] == [["fish", 0.7], ["crab", 0.2], ["rock", 0.1]]
    # A class that never wins still needs a category, or prob cannot name it.
    assert sorted(category["name"] for category in doc["categories"]) == [
        "crab", "fish", "rock"]
    # prob is positional over the categories table, which is only final at write time.
    ordered = [category["name"] for category in doc["categories"]]
    assert ann["prob"] == [dict(ann["confidence_pairs"])[name] for name in ordered]


@requires_kwiver
def test_top_n_classes_caps_the_pairs(tmp_path):
    """Mirrors the viame_csv writer's option of the same name; 0 keeps all."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    def run(top_n, name):
        writer = WriteDetectedObjectSetCoco()
        writer.set_configuration(writer.get_configuration())
        writer.top_n_classes = top_n
        out = str(tmp_path / name)
        writer.open(out)
        writer.write_set(vital_types.DetectedObjectSet([
            _multiclass_detection(1, 2, 3, 4,
                                  [("fish", 0.7), ("crab", 0.2), ("rock", 0.1)])]),
            "f000.png")
        writer.complete()
        writer.close()
        return _load(out)

    capped = run(2, "capped.json")
    _assert_profile(capped)
    assert capped["annotations"][0]["confidence_pairs"] == [
        ["fish", 0.7], ["crab", 0.2]]
    # A class that was cut earns no category, so prob stays aligned.
    assert sorted(c["name"] for c in capped["categories"]) == ["crab", "fish"]
    assert capped["annotations"][0]["prob"] == [0.7, 0.2]

    uncapped = run(0, "uncapped.json")
    assert len(uncapped["annotations"][0]["confidence_pairs"]) == 3


def test_confidence_pairs_recovered_from_either_spelling():
    ordered = ["fish", "crab"]
    exact = {"confidence_pairs": [["crab", 0.25], ["fish", 0.75]]}
    assert uc.confidence_pairs_from_annotation(exact, ordered) == [
        ("crab", 0.25), ("fish", 0.75)]
    legacy = {"dive_confidence_pairs": [["crab", 0.25], ["fish", 0.75]]}
    assert uc.confidence_pairs_from_annotation(legacy, ordered) == [
        ("crab", 0.25), ("fish", 0.75)]

    # Falls back to the positional vector when the sparse form is absent.
    assert uc.confidence_pairs_from_annotation({"prob": [0.75, 0.25]}, ordered) == [
        ("fish", 0.75), ("crab", 0.25)]

    # A prob vector that does not line up with the categories is unusable.
    assert uc.confidence_pairs_from_annotation({"prob": [0.75]}, ordered) == []
    assert uc.confidence_pairs_from_annotation({}, ordered) == []


@requires_kwiver
def test_info_block_carries_provenance(tmp_path):
    """MS-COCO info fields, written only when the caller supplies them."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    writer.version_identifier = "1.2.3"
    writer.contributor = "VIAME"

    out = str(tmp_path / "detections.json")
    writer.open(out)
    writer.write_set(vital_types.DetectedObjectSet(
        [_detection(1, 2, 3, 4, "fish")]), "f000.png")
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    assert doc["info"]["version"] == "1.2.3"
    assert doc["info"]["contributor"] == "VIAME"

    plain = _write_detections(tmp_path, "plain.json",
                              [("f000.png", [_detection(1, 2, 3, 4, "fish")])])
    assert "version" not in plain["info"]
    assert "contributor" not in plain["info"]


@requires_kwiver
def test_frame_rate_recorded_for_video(tmp_path):
    """The CSV header carries a frame rate; COCO has to carry it too."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    writer.frame_rate = "5"

    out = str(tmp_path / "clip.json")
    writer.open(out)
    writer.write_set(vital_types.DetectedObjectSet(
        [_detection(1, 2, 3, 4, "fish")]), "")
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc, expect_video=True)
    # Named for what it is: the rate annotations were produced at.
    assert [video["annotation_fps"] for video in doc["videos"]] == [5.0]
    assert "fps" not in doc["info"]


@requires_kwiver
def test_frame_rate_absent_for_image_lists(tmp_path):
    """A frame rate describes a video; an image list has none to describe."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    writer.frame_rate = "5"

    out = str(tmp_path / "detections.json")
    writer.open(out)
    writer.write_set(vital_types.DetectedObjectSet(
        [_detection(1, 2, 3, 4, "fish")]), "f000.png")
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    assert "videos" not in doc
    assert "annotation_fps" not in doc["info"]


@requires_kwiver
def test_frame_rate_absent_when_unset(tmp_path):
    doc = _write_detections(tmp_path, "clip.json", [("", [])], video_name="clip")
    assert not any("annotation_fps" in video for video in doc["videos"])


def test_unusable_frame_rates_are_dropped():
    for value in ("", "not-a-number", "0", "-1", None):
        assert uc._parse_fps(value) is None
    assert uc._parse_fps("29.97") == pytest.approx(29.97)


def test_multichannel_imagery_listed_as_assets():
    """kwcoco marks the older 'auxiliary' spelling as pending deprecation."""
    entries = uc.build_image_list(["img.png"], ["ir"], [".ir"])

    assert "auxiliary" not in entries[0]
    assert [asset["channels"] for asset in entries[0]["assets"]] == ["ir"]


@requires_kwiver
def test_detection_writer_video_name_overrides(tmp_path):
    frames = [("f000.png", [_detection(1, 2, 3, 4, "fish")])]
    doc = _write_detections(tmp_path, "detections.json", frames,
                            video_name="survey_07")

    _assert_profile(doc, expect_video=True)
    assert doc["videos"] == [{"id": 1, "name": "survey_07"}]


@requires_kwiver
def test_track_writer_image_list_has_no_video(tmp_path):
    """Named frames mean an image list, so no videos table and no video_id."""
    from viame.core.write_object_track_set_coco import WriteObjectTrackSetCoco

    track = vital_types.Track(id=1)
    for frame in (0, 1):
        track.append(vital_types.ObjectTrackState(
            frame, frame * 1000000, _detection(10, 10, 20, 20, "fish")))

    writer = WriteObjectTrackSetCoco()
    writer.set_configuration(writer.get_configuration())

    out = str(tmp_path / "tracks.json")
    writer.open(out)
    for frame, name in [(0, "f000.png"), (1, "f001.png")]:
        stamp = vital_types.Timestamp()
        stamp.set_frame(frame)
        writer.write_set(vital_types.ObjectTrackSet([track]), stamp, name)
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    assert "videos" not in doc
    assert not any("video_id" in image for image in doc["images"])
    assert [image["file_name"] for image in doc["images"]] == ["f000.png", "f001.png"]


@requires_kwiver
def test_attributes_round_trip_under_their_own_key(tmp_path):
    """Attributes belong under `attributes`, not scattered at the top level."""
    from viame.core.write_detected_object_set_coco import WriteDetectedObjectSetCoco

    det = _detection(1, 2, 3, 4, "fish")
    det.add_note(json.dumps({"occluded": True, "track_attributes": {"gear": "trawl"}}))
    det.add_note("a plain note")

    writer = WriteDetectedObjectSetCoco()
    writer.set_configuration(writer.get_configuration())
    out = str(tmp_path / "detections.json")
    writer.open(out)
    writer.write_set(vital_types.DetectedObjectSet([det]), "f000.png")
    writer.complete()
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    ann = doc["annotations"][0]
    assert ann["attributes"] == {"occluded": True}
    # A kwiver detection has no track-level store, so this rides as a note.
    assert ann["track_attributes"] == {"gear": "trawl"}
    assert ann["notes"] == ["a plain note"]
    assert "occluded" not in ann

    restored = uc.annotation_to_detection(ann, {1: "fish"})
    carried = {}
    plain = []
    for note in restored.notes:
        try:
            carried.update(json.loads(note))
        except ValueError:
            plain.append(note)
    assert carried == {"occluded": True, "track_attributes": {"gear": "trawl"}}
    assert plain == ["a plain note"]


@requires_kwiver
def test_track_writer_counts_frames_without_a_timestamp(tmp_path):
    """Conversion pipelines have no frame clock, so the writer keeps its own."""
    from viame.core.write_object_track_set_coco import WriteObjectTrackSetCoco

    track = vital_types.Track(id=3)
    for frame in (0, 1):
        track.append(vital_types.ObjectTrackState(
            frame, 0, _detection(10, 10, 20, 20, "fish")))

    writer = WriteObjectTrackSetCoco()
    writer.set_configuration(writer.get_configuration())
    out = str(tmp_path / "tracks.json")
    writer.open(out)
    for name in ("a.png", "b.png"):
        # An invalid timestamp is what a reader-driven pipeline supplies.
        writer.write_set(vital_types.ObjectTrackSet([track]),
                         vital_types.Timestamp(), name)
    writer.close()

    doc = _load(out)
    _assert_profile(doc)
    # Names still land on the right frames, and no videos table is invented.
    assert [(image["frame_index"], image["file_name"]) for image in doc["images"]] == [
        (0, "a.png"), (1, "b.png")]
    assert "videos" not in doc
