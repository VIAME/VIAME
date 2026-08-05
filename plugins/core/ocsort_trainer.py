# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT / Deep OC-SORT tracker training implementation.

OC-SORT uses Kalman filtering with observation-centric improvements (no
learned components), so "training" primarily estimates optimal parameters
from groundtruth tracks:
- Kalman filter noise weights, from observed motion patterns
- Detection confidence thresholds and track buffer
- The IoU association gates of each matching stage, from the overlaps the
  groundtruth's own consecutive-frame links present to them
- Velocity direction consistency (VDC) weight
- Observation-centric velocity window (delta_t)

When ``use_reid`` is enabled (Deep OC-SORT), an appearance Re-ID model is
additionally trained, reusing the BoT-SORT Re-ID training recipe. That
path is imported lazily so a default (motion-only) run pulls in no
torch/pytorch dependency.

The emitted parameters are consumed by the ocsort track_objects
implementation via its ``params_file`` (and, for Deep OC-SORT, its
``model_path``) configuration values.
"""

import os
import json
import numpy as np

from viame.core.training_data import detector_statistics

from kwiver.vital.algo import TrainTracker


# Competitors scored against a single link when measuring how well the
# velocity direction penalty separates a true candidate from a wrong one.
# Only their mean is used, so a crowded frame is strided rather than
# enumerated, which keeps the measurement linear in the annotation count.
_CROSS_SAMPLE_LIMIT = 32


def _box_iou( a, b ):
    """Intersection over union of two ( x1, y1, x2, y2 ) boxes."""
    x0 = max( a[ 0 ], b[ 0 ] )
    y0 = max( a[ 1 ], b[ 1 ] )
    x1 = min( a[ 2 ], b[ 2 ] )
    y1 = min( a[ 3 ], b[ 3 ] )

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = ( x1 - x0 ) * ( y1 - y0 )
    area_a = ( a[ 2 ] - a[ 0 ] ) * ( a[ 3 ] - a[ 1 ] )
    area_b = ( b[ 2 ] - b[ 0 ] ) * ( b[ 3 ] - b[ 1 ] )

    return inter / ( area_a + area_b - inter )


def _box_center( b ):
    return np.array( [ ( b[ 0 ] + b[ 2 ] ) / 2.0, ( b[ 1 ] + b[ 3 ] ) / 2.0 ] )


def _shift_box( b, offset ):
    return ( b[ 0 ] + offset[ 0 ], b[ 1 ] + offset[ 1 ],
             b[ 2 ] + offset[ 0 ], b[ 3 ] + offset[ 1 ] )


def _vdc_factor( velocity, residual ):
    """OC-SORT's velocity direction penalty, before vdc_weight scales it.

    Mirrors velocity_direction_consistency in the tracker: zero unless the
    step an association would require points backwards along the track's
    observed velocity, rising to 2 for a full reversal.
    """
    speed = np.linalg.norm( velocity )
    length = np.linalg.norm( residual )

    if speed < 1e-5 or length < 1e-5:
        return 0.0

    cosine = float( np.dot( velocity / speed, residual / length ) )

    return ( 1.0 - cosine ) if cosine < 0 else 0.0


class OCSORTTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for OC-SORT parameter estimation
    and optional Deep OC-SORT Re-ID model training.
    """
    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-ocsort-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "ocsort_tracker"
        self._pipeline_template = ""
        self._threshold = "0.00"

        # Directory of detector output for the same clips, one VIAME CSV
        # per clip named after it. Left empty the estimates come from the
        # groundtruth alone, whose confidences are all 1.0, so the
        # confidence thresholds carry no information about the detector
        # this tracker will actually run behind.
        self._computed_detections = ""
        # Written by the training tool: which frames of the flat list belong
        # to which track set. Without it the detector statistics fall back to
        # guessing from the directory layout, which fails open the moment the
        # image tree holds more directories than there are annotated clips.
        self._sequence_manifest = ""
        self._delta_t = "3"

        # The IoU association gates are fit so that this fraction of the
        # groundtruth's true consecutive-frame links clears them. 99.5
        # follows ByteTrack: assignment still picks the best candidate among
        # those admitted, so a gate is a safety net against dropping a true
        # link rather than a discriminator between candidates, and a tight
        # quantile measured on one split does not transfer to another. The
        # floor and ceiling bound the gate itself (an IoU), not the emitted
        # match_thresh values, which are 1 - gate.
        self._match_gate_admit_percent = 99.5
        self._min_match_gate = 0.02
        self._max_match_gate = 0.5

        # Observation-centric recovery is fit far tighter, and against a
        # different quantity: see _estimate_ocr_gate.
        self._ocr_gate_admit_percent = 50.0
        self._min_ocr_gate = 0.1
        self._max_ocr_gate = 0.5

        # Deep OC-SORT (appearance Re-ID) training. Disabled by default;
        # when off, no torch/pytorch code is imported.
        self._use_reid = "false"
        self._crop_size = "128x64"
        self._embedding_dim = "512"
        self._backbone = "resnet18"
        self._max_epochs = "50"
        self._batch_size = "32"
        self._learning_rate = "0.0003"
        self._gpu_count = -1

        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()

        cfg.set_value("identifier", self._identifier)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("output_directory", self._output_directory)
        cfg.set_value("output_prefix", self._output_prefix)
        cfg.set_value("pipeline_template", self._pipeline_template)
        cfg.set_value("threshold", self._threshold)
        cfg.set_value("computed_detections", self._computed_detections)
        cfg.set_value("sequence_manifest", self._sequence_manifest)
        cfg.set_value("delta_t", self._delta_t)
        cfg.set_value("match_gate_admit_percent",
                      str(self._match_gate_admit_percent))
        cfg.set_value("min_match_gate", str(self._min_match_gate))
        cfg.set_value("max_match_gate", str(self._max_match_gate))
        cfg.set_value("ocr_gate_admit_percent",
                      str(self._ocr_gate_admit_percent))
        cfg.set_value("min_ocr_gate", str(self._min_ocr_gate))
        cfg.set_value("max_ocr_gate", str(self._max_ocr_gate))
        cfg.set_value("use_reid", self._use_reid)
        cfg.set_value("crop_size", self._crop_size)
        cfg.set_value("embedding_dim", self._embedding_dim)
        cfg.set_value("backbone", self._backbone)
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("gpu_count", str(self._gpu_count))

        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._output_directory = str(cfg.get_value("output_directory"))
        self._output_prefix = str(cfg.get_value("output_prefix"))
        self._pipeline_template = str(cfg.get_value("pipeline_template"))
        self._threshold = str(cfg.get_value("threshold"))
        self._computed_detections = str(cfg.get_value("computed_detections"))
        self._sequence_manifest = str(cfg.get_value("sequence_manifest"))
        self._delta_t = str(cfg.get_value("delta_t"))
        self._match_gate_admit_percent = \
            float(cfg.get_value("match_gate_admit_percent"))
        self._min_match_gate = float(cfg.get_value("min_match_gate"))
        self._max_match_gate = float(cfg.get_value("max_match_gate"))
        self._ocr_gate_admit_percent = \
            float(cfg.get_value("ocr_gate_admit_percent"))
        self._min_ocr_gate = float(cfg.get_value("min_ocr_gate"))
        self._max_ocr_gate = float(cfg.get_value("max_ocr_gate"))
        self._use_reid = str(cfg.get_value("use_reid"))
        self._crop_size = str(cfg.get_value("crop_size"))
        self._embedding_dim = str(cfg.get_value("embedding_dim"))
        self._backbone = str(cfg.get_value("backbone"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._gpu_count = int(cfg.get_value("gpu_count"))

        if self._train_directory:
            if not os.path.exists(self._train_directory):
                os.makedirs(self._train_directory)

        if self._output_directory:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or \
          len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def _reid_enabled(self):
        return str(self._use_reid).lower() in ('true', '1', 'yes')

    def add_data_from_disk(self, categories, train_files, train_tracks,
                           test_files, test_tracks):
        print("Adding training data from disk...")
        print("  Training files: ", len(train_files))
        print("  Training tracks: ", len(train_tracks))
        print("  Test files: ", len(test_files))
        print("  Test tracks: ", len(test_tracks))

        if categories is not None:
            self._categories = categories.all_class_names()
        else:
            self._categories = []

        self._train_image_files = list(train_files)
        self._train_tracks = list(train_tracks)
        self._test_image_files = list(test_files)
        self._test_tracks = list(test_tracks)

    def _extract_track_statistics(self):
        """
        Extract statistics including velocity direction changes for VDC tuning.
        """
        positions = []
        velocities = []
        confidences = []
        track_lengths = []
        gap_lengths = []
        direction_changes = []  # Angle changes in velocity direction

        # Boxes per track, kept per clip so that a candidate can later be
        # scored against the other animals it actually competes with. The
        # association gates are fit from these in a second pass, once
        # delta_t is known, since the window decides which past observation
        # the velocity direction is measured from.
        clips = []

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            clip = []
            clips.append(clip)

            for track in track_set.tracks():
                states = list(track)
                track_lengths.append(len(states))

                observations = []
                clip.append(observations)

                prev_frame = None
                prev_cx, prev_cy = None, None
                prev_vx, prev_vy = None, None

                for state in states:
                    frame_id = state.frame_id
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box
                    x1 = bbox.min_x()
                    y1 = bbox.min_y()
                    x2 = bbox.max_x()
                    y2 = bbox.max_y()
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    positions.append((cx, cy, w, h))
                    observations.append((frame_id, (x1, y1, x2, y2)))

                    if det.confidence is not None:
                        confidences.append(det.confidence)

                    if prev_frame is not None and prev_cx is not None:
                        dt = frame_id - prev_frame
                        if dt > 0:
                            vx = (cx - prev_cx) / dt
                            vy = (cy - prev_cy) / dt
                            velocities.append((vx, vy, h, dt))

                            # Track velocity direction changes
                            if prev_vx is not None and dt == 1:
                                v1 = np.array([prev_vx, prev_vy])
                                v2 = np.array([vx, vy])
                                n1 = np.linalg.norm(v1)
                                n2 = np.linalg.norm(v2)

                                if n1 > 1e-5 and n2 > 1e-5:
                                    cos_angle = np.dot(v1, v2) / (n1 * n2)
                                    cos_angle = np.clip(cos_angle, -1, 1)
                                    angle = np.arccos(cos_angle)
                                    direction_changes.append(angle)

                            if dt > 1:
                                gap_lengths.append(dt - 1)

                            prev_vx, prev_vy = vx, vy

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy

                observations.sort()

        return {
            'positions': positions,
            'velocities': velocities,
            'confidences': confidences,
            'track_lengths': track_lengths,
            'gap_lengths': gap_lengths,
            'direction_changes': direction_changes,
            'clips': clips
        }

    def _association_statistics( self, clips, delta_t ):
        """What each of OC-SORT's association gates actually sees.

        OC-SORT does not compare the same pair of boxes at every stage, so
        one distribution of overlaps is not enough to fit all of its gates
        from. Four quantities are measured here, each matching the boxes a
        particular gate is handed:

        raw_link_ious
            Previous observation against the next one. An unconfirmed track
            has been seen once and its Kalman state was initiated with zero
            velocity, so nothing is compensated and its prediction is the
            previous box itself.

        pred_link_ious
            A constant-velocity step off the previous box against the next
            observation. This is what a confirmed track's gate sees: the
            filter has converged on the last observed displacement, so most
            of the motion is already removed before the IoU is taken. On
            FishTrack23 this is a visibly easier gate than the raw overlap
            ByteTrack fits against -- 5th percentile 0.211 against 0.072 --
            which is exactly why the ByteTrack quantiles cannot simply be
            copied across.

        gap_reobs_ious
            Last box before an annotation gap against the first box after
            it. Observation-centric recovery matches a lost track's *stale*
            last observation, with no prediction of any kind, so this and
            not either link distribution is what its gate must admit.

        link_vdc_factors / cross_vdc_*
            The velocity direction penalty the tracker would attach to a
            true link, and to the same track paired with the other animals
            present on that frame. Stage one adds this to the IoU distance
            before the gate sees it, so the gate cannot be fit without it,
            and the two together say whether the term separates a true
            candidate from a wrong one at all. Only the mean of the
            cross-animal side is ever read, so it is accumulated rather than
            kept, and a crowded frame is strided through: the number of
            pairs is the number of links times the number of animals in
            frame, which is the one quantity here that grows quadratically.
        """
        raw_link_ious = []
        pred_link_ious = []
        gap_reobs_ious = []
        link_vdc_factors = []
        cross_vdc_total = 0.0
        cross_vdc_count = 0

        for clip in clips:
            by_frame = {}

            for observations in clip:
                for frame, box in observations:
                    by_frame.setdefault( frame, [] ).append( box )

            for observations in clip:
                frames = dict( observations )

                for index in range( 1, len( observations ) ):
                    frame, box = observations[ index ]
                    prev_frame, prev_box = observations[ index - 1 ]
                    step = frame - prev_frame

                    if step != 1:
                        if step > 1:
                            gap_reobs_ious.append( _box_iou( prev_box, box ) )
                        continue

                    raw_link_ious.append( _box_iou( prev_box, box ) )

                    if prev_frame - 1 not in frames:
                        continue

                    velocity = _box_center( prev_box ) - \
                        _box_center( frames[ prev_frame - 1 ] )
                    predicted = _shift_box( prev_box, velocity )
                    predicted_center = _box_center( predicted )

                    pred_link_ious.append( _box_iou( predicted, box ) )

                    # Observation-centric momentum takes the oldest
                    # observation inside the delta_t window, exactly as
                    # STrack._refresh_velocity walks it.
                    momentum = None
                    span = 1

                    for back in range( delta_t, 0, -1 ):
                        if prev_frame - back in frames:
                            momentum = frames[ prev_frame - back ]
                            span = back
                            break

                    if momentum is None:
                        continue

                    momentum = ( _box_center( prev_box ) -
                                 _box_center( momentum ) ) / span

                    link_vdc_factors.append( _vdc_factor(
                        momentum, _box_center( box ) - predicted_center ) )

                    competitors = by_frame.get( frame, [] )
                    stride = 1 + len( competitors ) // _CROSS_SAMPLE_LIMIT

                    for other in competitors[ ::stride ]:
                        if other is box:
                            continue

                        cross_vdc_total += _vdc_factor(
                            momentum,
                            _box_center( other ) - predicted_center )
                        cross_vdc_count += 1

        return {
            'raw_link_ious': raw_link_ious,
            'pred_link_ious': pred_link_ious,
            'gap_reobs_ious': gap_reobs_ious,
            'link_vdc_factors': link_vdc_factors,
            'cross_vdc_total': cross_vdc_total,
            'cross_vdc_count': cross_vdc_count
        }

    def _estimate_kalman_parameters(self, stats):
        """Estimate Kalman filter parameters (same as ByteTrack)."""
        velocities = stats['velocities']

        if len(velocities) < 10:
            return 1.0 / 20, 1.0 / 160

        pos_variances = []
        for vx, vy, h, dt in velocities:
            if h > 0 and dt == 1:
                pos_var = np.sqrt(vx**2 + vy**2) / h
                pos_variances.append(pos_var)

        if len(pos_variances) > 0:
            median_pos_var = np.median(pos_variances)
            std_weight_position = np.clip(median_pos_var * 2, 0.01, 0.5)
        else:
            std_weight_position = 1.0 / 20

        vel_variances = []
        prev_vx, prev_vy, prev_h = None, None, None
        for vx, vy, h, dt in velocities:
            if prev_vx is not None and dt == 1 and h > 0:
                ax = vx - prev_vx
                ay = vy - prev_vy
                vel_var = np.sqrt(ax**2 + ay**2) / h
                vel_variances.append(vel_var)
            prev_vx, prev_vy, prev_h = vx, vy, h

        if len(vel_variances) > 0:
            median_vel_var = np.median(vel_variances)
            std_weight_velocity = np.clip(median_vel_var * 2, 0.001, 0.1)
        else:
            std_weight_velocity = 1.0 / 160

        return std_weight_position, std_weight_velocity

    def _detector_stats( self ):
        """Measure the detector against the groundtruth, when one was given."""
        if not self._computed_detections:
            return None

        stats = detector_statistics(
            self._train_tracks + self._test_tracks,
            self._train_image_files + self._test_image_files,
            self._computed_detections,
            sequence_manifest=self._sequence_manifest )

        matched = len( stats[ 'matched_confidences' ] )
        unmatched = len( stats[ 'unmatched_confidences' ] )

        print( "Computed detections: {} matched a groundtruth box, {} did "
               "not, over {} of {} annotated frames".format(
                   matched, unmatched, stats[ 'frames_with_computed' ],
                   stats[ 'frames_total' ] ) )

        if stats[ 'frames_total' ] and \
                stats[ 'frames_with_computed' ] < 0.1 * stats[ 'frames_total' ]:
            print( "WARNING: almost no annotated frame has a computed "
                   "detection. Frame ids here are positions within a clip, so "
                   "the detections must come from a run over the same "
                   "extracted frames rather than over the source video. "
                   "Falling back to the groundtruth." )
            return None

        if matched < 10 or unmatched < 10:
            print( "Too few matched or unmatched detections to separate the "
                   "two, falling back to the groundtruth." )
            return None

        return stats

    def _estimate_thresholds_from_detector( self, stats ):
        """Thresholds that separate the detector's hits from its misfires.

        The groundtruth cannot answer this: every annotation is confidence
        1.0, so a percentile of it is a percentile of a constant and the
        result is whatever the clamp allows. What a threshold has to do is
        divide the scores of detections that found a real object from the
        scores of those that did not, and that needs the detector's own
        output.
        """
        matched = np.array( stats[ 'matched_confidences' ] )
        unmatched = np.array( stats[ 'unmatched_confidences' ] )

        # Keep most real detections in the high tier
        high_thresh = float( np.percentile( matched, 25 ) )

        # The low tier is ByteTrack's second chance: it should reach well
        # below the high tier without swallowing the bulk of the misfires
        low_thresh = float( max( np.percentile( matched, 2 ),
                                 np.percentile( unmatched, 60 ) ) )

        # Starting a track off a misfire costs more than missing one, so this
        # sits above the high tier
        new_track_thresh = float( np.percentile( matched, 40 ) )

        high_thresh = float( np.clip( high_thresh, 0.05, 0.95 ) )
        low_thresh = float( np.clip( low_thresh, 0.01, high_thresh - 0.05 ) )
        new_track_thresh = float( np.clip( new_track_thresh,
                                           high_thresh, 0.95 ) )

        print( "  thresholds from the detector: high {:.3f} low {:.3f} "
               "new_track {:.3f}".format( high_thresh, low_thresh,
                                          new_track_thresh ) )

        return high_thresh, low_thresh, new_track_thresh

    def _measurement_noise_from_detector( self, stats ):
        """Localisation error of the detector, for the Kalman update.

        Estimated from how far a detection sits from the box it matched,
        which is the quantity the filter's measurement noise describes. The
        groundtruth cannot supply it: measured against itself its error is
        zero.
        """
        errors = stats[ 'center_errors' ]

        if len( errors ) < 10:
            return None

        return float( np.median( errors ) )

    def _estimate_thresholds(self, stats):
        """Estimate detection confidence thresholds."""
        confidences = stats['confidences']

        if len(confidences) < 10:
            return 0.6, 0.1, 0.6

        confidences = np.array(confidences)

        high_thresh = np.percentile(confidences, 30)
        low_thresh = np.percentile(confidences, 10)
        new_track_thresh = high_thresh

        high_thresh = np.clip(high_thresh, 0.3, 0.9)
        low_thresh = np.clip(low_thresh, 0.05, high_thresh - 0.1)
        new_track_thresh = np.clip(new_track_thresh, 0.3, 0.9)

        return float(high_thresh), float(low_thresh), float(new_track_thresh)

    def _estimate_track_buffer(self, stats):
        """Estimate track buffer from gap statistics."""
        gap_lengths = stats['gap_lengths']

        if len(gap_lengths) < 5:
            return 30

        gap_90 = np.percentile(gap_lengths, 90)
        track_buffer = int(gap_90 * 1.5) + 5
        track_buffer = max(10, min(100, track_buffer))

        return track_buffer

    def _estimate_delta_t(self, stats):
        """
        Estimate the observation-centric velocity window (OCM delta_t).

        A window spanning the typical annotation gap keeps velocity
        estimated from a real prior observation across short misses.
        """
        gap_lengths = stats['gap_lengths']
        if len(gap_lengths) < 5:
            return int(self._delta_t)
        return int(np.clip(np.percentile(gap_lengths, 75) + 1, 1, 5))

    def _estimate_vdc_weight(self, stats):
        """
        Estimate velocity direction consistency weight.

        Higher weight is warranted when motion is mostly linear (so a
        direction reversal is strong evidence against an association);
        lower when motion is erratic.

        That is the motion argument, and it is only half of it. The penalty
        is added straight onto the IoU distance that stage one gates on, so
        every point of weight is spent out of the same budget the gate has,
        and it is only worth spending where the penalty is actually larger
        for a wrong candidate than for a true one. The groundtruth answers
        that directly, by scoring the term against the animal a track really
        continues into and against every other animal on the same frame, so
        the motion weight is scaled by the separation between the two.

        On FishTrack23 that separation is *negative* (mean factor 0.94 on
        true links against 0.90 on cross-animal candidates, both non-zero on
        about 53% of pairs), and the reason is visible in the tracker: the
        direction it tests is the direction from the Kalman *prediction* to
        the candidate, and on a track the filter is predicting well that
        residual is noise, pointing backwards about half the time whoever
        the candidate is. Left at the stock 0.2 the term vetoes 7.5% of true
        links at the fitted gate and separates nothing, so the fit takes it
        to zero and the tracker skips the stage entirely. Over a handful of
        clips the separation is a small number of either sign, which is the
        same answer: a term that cannot be shown to discriminate is not
        worth much of the gate's budget.
        """
        direction_changes = stats['direction_changes']

        if len(direction_changes) < 10:
            return 0.2  # Default

        direction_changes_deg = np.array(direction_changes) * 180 / np.pi
        mean_change = np.mean(direction_changes_deg)
        std_change = np.std(direction_changes_deg)

        if mean_change < 20:
            vdc_weight = 0.3
        elif mean_change < 45:
            vdc_weight = 0.2
        else:
            vdc_weight = 0.1

        print(f"  Mean direction change: {mean_change:.1f} degrees")
        print(f"  Std direction change: {std_change:.1f} degrees")

        link_factors = stats.get( 'link_vdc_factors', [] )
        cross_count = stats.get( 'cross_vdc_count', 0 )

        if len( link_factors ) < 100 or cross_count < 100:
            return vdc_weight

        separation = float( stats[ 'cross_vdc_total' ] / cross_count -
                            np.mean( np.array( link_factors ) ) )

        # Scaled by the separation in factor units. A full reversal is 2, so
        # a separation of 1 is a wrong candidate penalised half a reversal
        # more than a true one on average -- ample evidence, and the point at
        # which the motion weight is kept whole.
        vdc_weight = round(
            vdc_weight * float( np.clip( separation, 0.0, 1.0 ) ), 3 )

        print( "  vdc penalty separates cross-animal from true candidates "
               "by {:+.3f} of a reversal -> vdc_weight {:.3f}".format(
                   separation / 2.0, vdc_weight ) )

        return vdc_weight

    def _estimate_match_thresholds( self, stats, vdc_weight ):
        """The IoU association gates, fit from the groundtruth's own links.

        A gate has one job: admit the overlap between where a track's animal
        is predicted to be and where it turns out to be, while excluding the
        overlap it has with the other animals competing for the same
        detection. The groundtruth supplies both distributions, and on
        FishTrack23 they barely meet -- against a track's prediction, fewer
        than 5% of cross-animal candidates reach even 0.02 IoU, while 98% of
        true links clear it.

        So the quantile is loose, and for the same reason ByteTrack's is:
        assignment still picks the best candidate among those admitted, so
        the gate is a safety net against dropping a true link rather than a
        discriminator between candidates, and a tight quantile fit on one
        split does not transfer to another.

        What does not carry across from ByteTrack is which overlap to
        measure. A confirmed track is gated on its velocity-compensated
        prediction, an unconfirmed one on its previous box, and those are
        different distributions -- fitting both from the raw link overlap is
        what leaves a brand-new track, the one with the worst prediction in
        the tracker, facing the strictest gate.

        The returned values are distance bounds (linear_assignment runs on
        1 - IoU), so LOOSENING a gate RAISES them.
        """
        stock = ( 0.8, 0.5, 0.7 )

        pred_ious = stats.get( 'pred_link_ious', [] )
        raw_ious = stats.get( 'raw_link_ious', [] )

        if len( pred_ious ) < 100 or len( raw_ious ) < 100:
            print( "Warning: too few consecutive-frame links ({}), keeping "
                   "the association gate defaults".format( len( raw_ious ) ) )
            return stock

        def gate( values ):
            fitted = float( np.percentile(
                np.array( values ), 100.0 - self._match_gate_admit_percent ) )
            return float( np.clip( fitted, self._min_match_gate,
                                   self._max_match_gate ) )

        pred_gate = gate( pred_ious )
        raw_gate = gate( raw_ious )

        # Stage one adds the velocity direction penalty to the IoU distance
        # before the gate sees it, so a true link that attracts one needs
        # that much more room. The allowance is capped rather than simply
        # added: at 1.0 a bound admits pairs that do not overlap at all, at
        # which point it has stopped being a gate, and the fix for a penalty
        # that will not fit under that ceiling is a smaller vdc_weight
        # rather than a gate that admits everything.
        factors = stats.get( 'link_vdc_factors', [] )
        allowance = 0.0

        if vdc_weight > 0 and factors:
            allowance = vdc_weight * float( np.percentile(
                np.array( factors ), self._match_gate_admit_percent ) )

        match_thresh = min( 1.0 - pred_gate + allowance,
                            1.0 - self._min_match_gate )

        # The second (low confidence) pass compares the same predictions
        # against the same kind of link, only with weaker detections, so it
        # is fit from the same distribution -- without the allowance, since
        # that pass costs on IoU alone.
        second_match_thresh = 1.0 - pred_gate

        # An unconfirmed track predicts nothing; its gate sees the raw
        # frame-to-frame overlap.
        unconfirmed_match_thresh = 1.0 - raw_gate

        print( "  gate IoU>={:.3f} on velocity-compensated predictions "
               "admits {:.1f}% of {} links".format(
                   pred_gate,
                   100 * float( np.mean( np.array( pred_ious ) >= pred_gate ) ),
                   len( pred_ious ) ) )
        print( "  gate IoU>={:.3f} on raw links admits {:.1f}% of "
               "{}".format(
                   raw_gate,
                   100 * float( np.mean( np.array( raw_ious ) >= raw_gate ) ),
                   len( raw_ious ) ) )

        return ( round( match_thresh, 3 ),
                 round( second_match_thresh, 3 ),
                 round( unconfirmed_match_thresh, 3 ) )

    def _estimate_ocr_gate( self, stats ):
        """The observation-centric recovery gate.

        Fit from a different quantity and at a far tighter quantile than the
        matching gates, because it is a different kind of gate. OCR compares
        a lost track's last real observation, stale by up to track_buffer
        frames, against detections nothing else claimed; there is no
        prediction to compensate the motion and no second opinion behind it.
        A wrong match here does not cost a frame of association, it
        transplants an identity and then replays that identity backwards
        through the Kalman filter as ORU interpolates the gap. A rejected
        true one only starts a new track. So this gate is a discriminator,
        and unlike the matching gates it cannot afford to err loose.

        It also could not be loosened usefully even if it were free: on
        FishTrack23 a quarter of the boxes on the far side of an annotation
        gap do not overlap their own predecessor at all, so no positive gate
        recovers them, and reaching for them means admitting every leftover
        detection in the frame. The gate is set at the median instead,
        keeping the half of re-observations that still overlap enough to be
        recognised as the same animal.
        """
        gap_ious = stats.get( 'gap_reobs_ious', [] )

        # Gaps are two orders of magnitude rarer than links, so the sample
        # floor is correspondingly lower; below it the percentile says more
        # about which clips were annotated than about the tracker.
        if len( gap_ious ) < 30:
            print( "Warning: only {} re-observations across an annotation "
                   "gap, keeping ocr_iou_thresh default".format(
                       len( gap_ious ) ) )
            return 0.3

        fitted = float( np.percentile(
            np.array( gap_ious ), 100.0 - self._ocr_gate_admit_percent ) )
        fitted = float( np.clip( fitted, self._min_ocr_gate,
                                 self._max_ocr_gate ) )

        print( "  ocr gate IoU>={:.3f} admits {:.1f}% of {} "
               "re-observations".format(
                   fitted,
                   100 * float( np.mean( np.array( gap_ious ) >= fitted ) ),
                   len( gap_ious ) ) )

        return round( fitted, 3 )

    def _train_reid_model(self):
        """
        Train the Deep OC-SORT appearance Re-ID model.

        Reuses the BoT-SORT Re-ID training recipe (imported lazily so torch
        is only pulled in when appearance training is requested). Returns
        the path to the trained model, or None on failure.
        """
        try:
            from viame.pytorch.botsort_trainer import BoTSORTTrainer
        except ImportError as e:
            print(f"[OCSORT] Re-ID training unavailable (pytorch plugin "
                  f"not importable): {e}")
            return None

        print("Training Deep OC-SORT appearance Re-ID model...")

        helper = BoTSORTTrainer()
        helper._train_directory = self._train_directory
        helper._crop_size = self._crop_size
        helper._embedding_dim = self._embedding_dim
        helper._backbone = self._backbone
        helper._max_epochs = self._max_epochs
        helper._batch_size = self._batch_size
        helper._learning_rate = self._learning_rate
        helper._gpu_count = self._gpu_count
        helper._train_image_files = self._train_image_files
        helper._train_tracks = self._train_tracks
        helper._test_image_files = self._test_image_files
        helper._test_tracks = self._test_tracks

        reid_dir = helper._prepare_reid_data()
        helper._train_reid_model(reid_dir)

        model_path = os.path.join(self._train_directory, "snapshot",
                                  "best_model.pth")
        if os.path.exists(model_path):
            return model_path

        print("[OCSORT] Re-ID training produced no model")
        return None

    def update_model(self):
        """
        Analyze track groundtruth and estimate OC-SORT parameters (and,
        for Deep OC-SORT, train the appearance Re-ID model).

        Returns:
            dict: Map of template replacements and file copies
        """
        print("Starting OC-SORT parameter estimation...")

        print("Extracting track statistics...")
        stats = self._extract_track_statistics()

        print(f"  Found {len(stats['positions'])} detections")
        print(f"  Found {len(stats['velocities'])} velocity measurements")
        print(f"  Found {len(stats['track_lengths'])} tracks")
        print(f"  Found {len(stats['direction_changes'])} direction changes")

        print("Estimating Kalman filter parameters...")
        std_weight_position, std_weight_velocity = self._estimate_kalman_parameters(stats)
        print(f"  std_weight_position: {std_weight_position:.6f}")
        print(f"  std_weight_velocity: {std_weight_velocity:.6f}")

        print("Estimating detection thresholds...")
        high_thresh, low_thresh, new_track_thresh = self._estimate_thresholds(stats)

        # A detector's own output, when supplied, answers what the
        # groundtruth cannot: where its boxes land relative to the truth, and
        # which of its scores correspond to real objects
        detector = self._detector_stats()

        if detector is not None:
            high_thresh, low_thresh, new_track_thresh = \
                self._estimate_thresholds_from_detector( detector )

            measured = self._measurement_noise_from_detector( detector )

            if measured is not None:
                std_weight_position = float( np.clip(
                    measured,
                    0.01, 0.5 ) )
                print( "  measurement noise from the detector: "
                       "std_weight_position {:.4f}".format(
                           std_weight_position ) )

        print(f"  high_thresh: {high_thresh:.3f}")
        print(f"  low_thresh: {low_thresh:.3f}")
        print(f"  new_track_thresh: {new_track_thresh:.3f}")

        print("Estimating track buffer...")
        track_buffer = self._estimate_track_buffer(stats)
        print(f"  track_buffer: {track_buffer}")

        delta_t = self._estimate_delta_t(stats)
        print(f"  delta_t: {delta_t}")

        # Measured after delta_t, since the observation-centric window
        # decides which past observation a velocity direction is taken from
        print("Measuring what the association gates see...")
        stats.update(self._association_statistics(stats['clips'], delta_t))
        print("  {} links, {} of them velocity compensated, {} "
              "re-observations across a gap".format(
                  len(stats['raw_link_ious']), len(stats['pred_link_ious']),
                  len(stats['gap_reobs_ious'])))

        print("Estimating VDC weight...")
        vdc_weight = self._estimate_vdc_weight(stats)
        print(f"  vdc_weight: {vdc_weight:.3f}")

        print("Estimating association gates...")
        match_thresh, second_match_thresh, unconfirmed_match_thresh = \
            self._estimate_match_thresholds(stats, vdc_weight)
        print(f"  match_thresh: {match_thresh:.3f}")
        print(f"  second_match_thresh: {second_match_thresh:.3f}")
        print(f"  unconfirmed_match_thresh: {unconfirmed_match_thresh:.3f}")

        ocr_iou_thresh = self._estimate_ocr_gate(stats)
        print(f"  ocr_iou_thresh: {ocr_iou_thresh:.3f}")

        use_reid = self._reid_enabled()

        params = {
            'std_weight_position': std_weight_position,
            'std_weight_velocity': std_weight_velocity,
            'high_thresh': high_thresh,
            'low_thresh': low_thresh,
            'match_thresh': match_thresh,
            'second_match_thresh': second_match_thresh,
            'unconfirmed_match_thresh': unconfirmed_match_thresh,
            'new_track_thresh': new_track_thresh,
            'track_buffer': track_buffer,
            'delta_t': delta_t,
            'vdc_weight': vdc_weight,
            'ocr_iou_thresh': ocr_iou_thresh,
            # A zero weight leaves the stage computing a matrix of zeros for
            # every track against every detection, so it is skipped outright
            'use_vdc': vdc_weight > 0,
            'use_oru': True,
            'use_byte': True,
            'use_ocr': True,
            'use_reid': use_reid,
        }

        # Save parameter JSON to train directory (copied out by the caller)
        params_file = os.path.join(self._train_directory, "ocsort_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {params_file}")

        # Optional Deep OC-SORT appearance model
        reid_model_path = None
        if use_reid:
            reid_model_path = self._train_reid_model()

        output = self._get_output_map(params, params_file, reid_model_path)

        print("\nOC-SORT parameter estimation complete!\n")

        return output

    def _get_output_map(self, params, params_file, reid_model_path=None):
        """Build output map with template replacements and file copies.

        Returns:
            dict: Map where file paths are file copies, other values are
            template replacements.
        """
        output = {}
        algo = "ocsort"

        output["type"] = algo

        # Config keys matching ocsort inference config
        for key, value in params.items():
            if isinstance(value, float):
                output[algo + ":" + key] = f"{value:.3f}"
            else:
                output[algo + ":" + key] = str(value)

        # File copies
        output["ocsort_params.json"] = params_file

        # Deep OC-SORT appearance model, if trained
        if reid_model_path is not None:
            output[algo + ":model_path"] = "ocsort_reid.pth"
            output["ocsort_reid.pth"] = reid_model_path

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "ocsort"

    if algorithm_factory.has_algorithm_impl_name(
        OCSORTTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "OC-SORT parameter estimation and optional Deep OC-SORT Re-ID training",
        OCSORTTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
