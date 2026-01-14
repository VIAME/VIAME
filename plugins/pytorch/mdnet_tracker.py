# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import scriptconfig as scfg
import numpy as np

from kwiver.vital.algo import TrackObjects

from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
    ObjectTrackState, ObjectTrackSet, Track,
)

import viame.pytorch.mdnet.tracker as mdnet


class MDNetTrackerConfig(scfg.DataConfig):
    """
    The configuration for :class:`MDNetTracker`.
    """
    weights_file = scfg.Value(
        'models/mdnet_seed.pth',
        help='MDNet initial weight file for each object track.')
    init_method = scfg.Value(
        'external_only',
        help='Method for initializing new tracks, can be: external_only or using_detections.')
    init_threshold = scfg.Value(
        0.20,
        help='If tracking multiple targets, the initialization threshold over '
             'detected object classifications for new tracks')
    iou_threshold = scfg.Value(
        0.50,
        help='If tracking multiple targets, the initialization threshold over '
             'box intersections in order to generate new tracks')
    type_string = scfg.Value(
        '',
        help='If non-empty set the output track to be a track of this object category.')

    def __post_init__(self):
        super().__post_init__()


class MDNetTracker(TrackObjects):
    """
    MDNet-based visual object tracker.

    This tracker uses deep learning for visual object tracking with online
    model updates. It maintains appearance models that are updated during
    tracking.

    References:
        https://github.com/hyeonseobnam/py-MDNet
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._kwiver_config = MDNetTrackerConfig()
        self._trackers = dict()
        self._tracks = dict()
        self._track_init_frames = dict()
        self._last_frame_id = -1
        self._last_frame = None

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.pytorch.utilities import vital_config_update
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        # Convert numeric values
        self._kwiver_config.init_threshold = float(self._kwiver_config.init_threshold)
        self._kwiver_config.iou_threshold = float(self._kwiver_config.iou_threshold)

        # Load model only once across all tracks for speed
        mdnet.opts['model_path'] = mdnet.MDNet(self._kwiver_config.weights_file)

        return True

    def check_configuration(self, cfg):
        return True

    def _format_image(self, image):
        """Convert image to numpy array format suitable for MDNet."""
        if not isinstance(image, np.ndarray):
            img_npy = image.asarray().astype('uint8')
            # Greyscale to color image if necessary
            if len(np.shape(img_npy)) > 2 and np.shape(img_npy)[2] == 1:
                img_npy = img_npy[:, :, 0]
            if len(np.shape(img_npy)) == 2:
                img_npy = np.stack((img_npy,) * 3, axis=-1)
            return img_npy
        return image

    def track(self, timestamp, image_data, detected_object_set, initializations=None,
              recommendations=None, evaluation_requests=None):
        """
        Track objects in the current frame.

        Args:
            timestamp: The current frame timestamp
            image_data: The current frame image
            detected_object_set: Detections for the current frame
            initializations: External track initializations
            recommendations: External track recommendations
            evaluation_requests: External detections to evaluate

        Returns:
            ObjectTrackSet: The current track states
        """
        if not timestamp.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        frame_id = timestamp.get_frame()

        if initializations is None:
            initializations = ObjectTrackSet()
        if recommendations is None:
            recommendations = ObjectTrackSet()
        if detected_object_set is None:
            detected_object_set = DetectedObjectSet()

        # Handle new track external initialization
        init_track_pool = initializations.tracks()
        recc_track_pool = recommendations.tracks()
        init_track_ids = []
        img_used = False

        if len(init_track_pool) != 0 or len(self._trackers) != 0:
            img_npy = self._format_image(image_data)
            img_used = True

        for trk in init_track_pool:
            # Special case, initialize a track on a previous frame
            if trk[trk.last_frame].frame_id == self._last_frame_id and \
               (trk.id not in self._track_init_frames or
                self._track_init_frames[trk.id] < self._last_frame_id):
                tid = trk.id
                cbox = trk[trk.last_frame].detection().bounding_box()
                bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
                last_frame_npy = self._format_image(self._last_frame)
                self._trackers[tid] = mdnet.MDNetTracker(last_frame_npy, bbox)
                self._tracks[tid] = [ObjectTrackState(timestamp, cbox, 1.0)]
                self._track_init_frames[tid] = self._last_frame_id
            # This track has an initialization signal for the current frame
            elif trk[trk.last_frame].frame_id == frame_id:
                tid = trk.id
                cbox = trk[trk.last_frame].detection().bounding_box()
                bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
                self._trackers[tid] = mdnet.MDNetTracker(img_npy, bbox)
                self._tracks[tid] = [ObjectTrackState(timestamp, cbox, 1.0)]
                init_track_ids.append(tid)
                self._track_init_frames[tid] = frame_id

        # Update existing tracks
        for tid in self._trackers.keys():
            if tid in init_track_ids:
                continue  # Already processed (initialized) on frame
            # Check if there's a recommendation for the update
            recc_bbox = []
            for trk in recc_track_pool:
                if trk.id == tid and trk[trk.last_frame].frame_id == frame_id:
                    cbox = trk[trk.last_frame].detection().bounding_box()
                    recc_bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
                    break
            bbox, score = self._trackers[tid].update(img_npy, likely_bbox=recc_bbox)
            if score > mdnet.opts['success_thr']:
                cbox = BoundingBoxD(
                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                new_state = ObjectTrackState(timestamp, cbox, score)
                self._tracks[tid].append(new_state)

        # Output results
        output_tracks = ObjectTrackSet(
            [Track(tid, trk) for tid, trk in self._tracks.items()])

        self._last_frame_id = timestamp.get_frame()
        if img_used:
            self._last_frame = img_npy
        else:
            self._last_frame = image_data

        return output_tracks


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "mdnet"

    if algorithm_factory.has_algorithm_impl_name(
            MDNetTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, "MDNet visual object tracker", MDNetTracker)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
