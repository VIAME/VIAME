# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""Text query on every frame through a vision-language model served by Ollama."""

from viame.algo import ImageObjectDetector, RefineTracks
from viame.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
    ObjectTrackSet, ObjectTrackState, Track,
)

from viame.segmentation.interactive_vlm import detect_objects
from viame.utilities.utils import image_container_to_uint8_hwc, str2bool


class _OllamaVlmQuery:
    """Configuration and per-frame query shared by the detector and refiner."""

    def _init_query(self):
        self._model = "qwen3-vl:8b"
        self._text_query = "object"
        self._think = False
        self._max_new_objects = 50

    def _query_config(self, cfg):
        cfg.set_value("model", self._model)
        cfg.set_value("text_query", self._text_query)
        cfg.set_value("think", str(self._think))
        cfg.set_value("max_new_objects", str(self._max_new_objects))
        return cfg

    def _set_query_config(self, cfg):
        self._model = cfg.get_value("model")
        self._text_query = cfg.get_value("text_query").strip()
        self._think = str2bool(cfg.get_value("think"))
        self._max_new_objects = int(cfg.get_value("max_new_objects"))

    @staticmethod
    def _query_config_valid(cfg):
        return bool(cfg.get_value("model")) and bool(cfg.get_value("text_query").strip())

    def _detect(self, image_data):
        from PIL import Image

        image = Image.fromarray(image_container_to_uint8_hwc(image_data))
        found = detect_objects(image, self._text_query, self._model, self._think)
        return [
            DetectedObject(
                BoundingBoxD(*box), 1.0,
                DetectedObjectType(label or self._text_query, 1.0))
            for box, label in found[:self._max_new_objects]
        ]


class OllamaVlmDetector(_OllamaVlmQuery, ImageObjectDetector):
    """Per-frame detections, e.g. to feed a tracker."""

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._init_query()

    def get_configuration(self):
        return self._query_config(super(ImageObjectDetector, self).get_configuration())

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self._set_query_config(cfg)

    def check_configuration(self, cfg):
        return self._query_config_valid(cfg)

    def detect(self, image_data):
        output = DetectedObjectSet()
        for det in self._detect(image_data):
            output.add(det)
        return output


class OllamaVlmRefiner(_OllamaVlmQuery, RefineTracks):
    """
    Adds one single-frame track per object found. Existing tracks pass
    through unchanged unless ``replace_existing`` is set.
    """

    def __init__(self):
        RefineTracks.__init__(self)
        self._init_query()
        self._replace_existing = True
        self._used_ids = set()
        self._input_id_map = {}
        self._next_id = 1

    def get_configuration(self):
        cfg = self._query_config(super(RefineTracks, self).get_configuration())
        cfg.set_value("replace_existing", str(self._replace_existing))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self._set_query_config(cfg)
        self._replace_existing = str2bool(cfg.get_value("replace_existing"))

    def check_configuration(self, cfg):
        return self._query_config_valid(cfg)

    def _allocate_id(self):
        while self._next_id in self._used_ids:
            self._next_id += 1
        self._used_ids.add(self._next_id)
        return self._next_id

    def _resolve_input(self, track):
        # Input tracks can first appear after new tracks were numbered, so an
        # input ID already handed out is moved to a fresh one.
        if track.id not in self._input_id_map:
            self._input_id_map[track.id] = (
                self._allocate_id() if track.id in self._used_ids else track.id)
            self._used_ids.add(self._input_id_map[track.id])
        resolved = self._input_id_map[track.id]
        if resolved == track.id:
            return track
        moved = Track(id=resolved)
        for state in track:
            moved.append(state)
        return moved

    def refine(self, ts, image_data, tracks):
        output = [] if self._replace_existing else [
            self._resolve_input(track) for track in tracks.tracks()]
        for det in self._detect(image_data):
            track = Track(id=self._allocate_id())
            track.append(ObjectTrackState(ts.get_frame(), ts.get_time_usec(), det))
            output.append(track)
        return ObjectTrackSet(output)


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OllamaVlmDetector,
        "ollama_vlm",
        "Text query detections from a vision-language model served by Ollama",
    )
    register_vital_algorithm(
        OllamaVlmRefiner,
        "ollama_vlm",
        "Text query on every frame through a vision-language model served by Ollama",
    )
