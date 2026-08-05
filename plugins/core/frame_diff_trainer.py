# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Three-frame-difference detector "training".

Nothing is learned in the usual sense: the detector is threshold, morphology
and connected components, so training means measuring the motion signal on
annotated animals and picking the settings that separate them from background.

Three questions are asked of the groundtruth:

  separation  How far apart the three frames are taken. Slow animals leave no
              signal at separation 1 and fast ones smear at 5, so the right
              value depends on the frame rate and on what is being watched.
              Each candidate separation is scored end to end, because the
              motion image it produces has its own scale and so needs its own
              threshold -- one cannot be carried over from another separation.
              That coupling is strong enough to reverse the ranking: measured
              against an adaptive threshold separation 3 beat 1 across the
              board, but held at a fixed low threshold it floods (on
              FishTrack Test, ap50 0.047 -> 0.019 with 28% more detections),
              because a wider separation moves more pixels past the same bar.
              Hence the sweep rather than a default.

  threshold   For each groundtruth box, a high percentile of the motion image
              inside it, against the same statistic on background patches of the
              same size drawn from the same frame. The threshold reported is the
              one maximising true positive rate minus false positive rate.

  morphology  For a grid of opening and closing radii, the mask is built,
              cleaned, split into connected components and matched to the
              groundtruth.

The morphology and separation are chosen together against `objective`. Recall
answers "how many animals did we touch", which is the right question when the
output feeds a fusion that will re-rank it anyway. F1 also charges for false
alarms, which matters when the output is consumed directly: on cluttered
scenes the most permissive settings reach the highest recall while emitting
tens of thousands of vegetation blobs per clip, and recall alone cannot see
that cost.

Emits a pipeline configured with the result.
"""

import os

import numpy as np

from kwiver.vital.algo import TrainDetector


def _to_grey(image):
    arr = np.asarray(image)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def three_frame_diff(prev_img, cur_img, next_img):
    """min(|I_t - I_{t-k}|, |I_{t+k} - I_t|)"""
    a, b, c = _to_grey(prev_img), _to_grey(cur_img), _to_grey(next_img)
    return np.minimum(np.abs(b - a), np.abs(c - b))


def boxes_from_mask(mask, open_r, close_r, min_area, max_area, min_fill):
    import cv2
    m = mask.astype(np.uint8)
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or a > max_area:
            continue
        if w * h <= 0 or a / float(w * h) < min_fill:
            continue
        out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


PIPELINE_TEMPLATE = """# Three-frame difference detector, trained settings.

config _pipeline:_edge
  :capacity                                    5

config _scheduler
  :type                                        pythread_per_process

include common_default_input_with_downsampler.pipe

process detector_input
  :: image_filter
  :filter:type                                 vxl_convert_image
  :filter:vxl_convert_image:format             byte
  :filter:vxl_convert_image:force_three_channel  true

connect from downsampler.output_1
        to   detector_input.image

process motion
  :: detect_motion
  :algo:type                                            ocv_3frame_differencing
  :algo:ocv_3frame_differencing:frame_separation        {frame_separation}
  :algo:ocv_3frame_differencing:jitter_radius           1
  :algo:ocv_3frame_differencing:max_foreground_fract    0.15
  :algo:ocv_3frame_differencing:max_foreground_fract_thresh  10

connect from detector_input.image
        to   motion.image

process binarize
  :: image_filter
  :filter:type                                 vxl_threshold
  :filter:vxl_threshold:type                   absolute
  :filter:vxl_threshold:threshold              {threshold}

connect from motion.motion_heat_map
        to   binarize.image

process opening
  :: image_filter
  :filter:type                                 vxl_morphology
  :filter:vxl_morphology:morphology            open
  :filter:vxl_morphology:element_shape         disk
  :filter:vxl_morphology:kernel_radius         {open_radius}

connect from binarize.image
        to   opening.image

process closing
  :: image_filter
  :filter:type                                 vxl_morphology
  :filter:vxl_morphology:morphology            close
  :filter:vxl_morphology:element_shape         disk
  :filter:vxl_morphology:kernel_radius         {close_radius}

connect from opening.image
        to   closing.image

process mask_to_byte
  :: image_filter
  :filter:type                                 vxl_convert_image
  :filter:vxl_convert_image:format             byte
  :filter:vxl_convert_image:scale_factor       255.0

connect from closing.image
        to   mask_to_byte.image

process detector
  :: image_object_detector
  :detector:type                                      detect_heat_map
  :detector:detect_heat_map:threshold                 0
  :detector:detect_heat_map:force_bbox_width          -1
  :detector:detect_heat_map:force_bbox_height         -1
  :detector:detect_heat_map:min_area                  {min_area}
  :detector:detect_heat_map:max_area                  {max_area}
  :detector:detect_heat_map:min_fill_fraction         {min_fill}
  :detector:detect_heat_map:class_name                {class_name}

connect from mask_to_byte.image
        to   detector.image

process detector_writer
  :: detected_object_output
  :file_name                                   computed_detections.csv
  :writer:type                                 viame_csv

connect from detector.detected_object_set
        to   detector_writer.detected_object_set
connect from downsampler.output_2
        to   detector_writer.image_file_name
"""


class FrameDiffTrainer(TrainDetector):
    """
    Implementation of TrainDetector for three-frame-difference settings.
    """

    def __init__(self):
        TrainDetector.__init__(self)

        self._identifier = "viame-frame-diff-detector"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "frame_diff"
        self._class_name = "motion"

        self._frame_separation = "1,3"
        self._min_area = "100"
        self._max_area = "400000"
        self._min_fill = "0.05"

        # Frames sampled per clip. The estimate converges quickly and every
        # sample costs three image loads.
        self._max_samples = "200"
        # IoU at which a component counts as having found an animal.
        self._match_iou = "0.5"
        self._open_radii = "0,1,2,3"
        self._close_radii = "0,2,4,7"
        self._objective = "recall"

        self._train_files = []
        self._train_dets = []

    def get_configuration(self):
        cfg = super(TrainDetector, self).get_configuration()
        cfg.set_value("identifier", self._identifier)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("output_directory", self._output_directory)
        cfg.set_value("output_prefix", self._output_prefix)
        cfg.set_value("class_name", self._class_name)
        cfg.set_value("frame_separation", self._frame_separation)
        cfg.set_value("min_area", self._min_area)
        cfg.set_value("max_area", self._max_area)
        cfg.set_value("min_fill", self._min_fill)
        cfg.set_value("max_samples", self._max_samples)
        cfg.set_value("match_iou", self._match_iou)
        cfg.set_value("open_radii", self._open_radii)
        cfg.set_value("close_radii", self._close_radii)
        cfg.set_value("objective", self._objective)
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._output_directory = str(cfg.get_value("output_directory"))
        self._output_prefix = str(cfg.get_value("output_prefix"))
        self._class_name = str(cfg.get_value("class_name"))
        self._frame_separation = str(cfg.get_value("frame_separation"))
        self._min_area = str(cfg.get_value("min_area"))
        self._max_area = str(cfg.get_value("max_area"))
        self._min_fill = str(cfg.get_value("min_fill"))
        self._max_samples = str(cfg.get_value("max_samples"))
        self._match_iou = str(cfg.get_value("match_iou"))
        self._open_radii = str(cfg.get_value("open_radii"))
        self._close_radii = str(cfg.get_value("close_radii"))
        self._objective = str(cfg.get_value("objective")).strip().lower()

        for d in (self._train_directory, self._output_directory):
            if d and not os.path.exists(d):
                os.makedirs(d)
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or \
          len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_dets,
                           test_files, test_dets):
        # Test data is deliberately ignored: the settings are estimated from
        # training data alone.
        self._train_files = list(train_files)
        self._train_dets = list(train_dets)
        print("Added %d training frames" % len(self._train_files))

    def _boxes(self, det_set):
        out = []
        if det_set is None:
            return out
        for det in det_set:
            if det is None:
                continue
            b = det.bounding_box
            out.append((b.min_x(), b.min_y(), b.max_x(), b.max_y()))
        return out

    def _sample(self, sep, limit):
        """Motion images and groundtruth for a sample of annotated frames, at
        one separation, plus the in-box and background motion statistics the
        threshold is derived from."""
        import cv2

        files, dets = self._train_files, self._train_dets
        usable = [i for i in range(sep, len(files) - sep)
                  if i < len(dets) and self._boxes(dets[i])]
        if not usable:
            return [], None, None
        step = max(1, len(usable) // limit)
        usable = usable[::step][:limit]

        # seeded per separation so every candidate sees the same background
        # patches, making the comparison between separations a fair one
        rng = np.random.default_rng(0)
        sig, bkg, frames = [], [], []

        for i in usable:
            try:
                imgs = [cv2.imread(files[j], cv2.IMREAD_GRAYSCALE)
                        for j in (i - sep, i, i + sep)]
            except Exception:
                continue
            if any(im is None for im in imgs):
                continue
            mot = three_frame_diff(*imgs)
            h, w = mot.shape
            boxes = [b for b in self._boxes(dets[i])
                     if 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h]
            if not boxes:
                continue
            frames.append((mot, boxes))
            for (x1, y1, x2, y2) in boxes:
                patch = mot[int(y1):int(y2), int(x1):int(x2)]
                if patch.size:
                    sig.append(np.percentile(patch, 90))
                bw, bh = int(x2 - x1), int(y2 - y1)
                if w - bw > 1 and h - bh > 1:
                    rx, ry = rng.integers(0, w - bw), rng.integers(0, h - bh)
                    cand = (rx, ry, rx + bw, ry + bh)
                    if iou(cand, (x1, y1, x2, y2)) <= 0.1:
                        p = mot[ry:ry + bh, rx:rx + bw]
                        if p.size:
                            bkg.append(np.percentile(p, 90))

        if not sig or not bkg:
            return frames, None, None
        return frames, np.asarray(sig), np.asarray(bkg)

    @staticmethod
    def _threshold(sig, bkg):
        cands = np.unique(np.percentile(np.concatenate([sig, bkg]),
                                        np.linspace(1, 99, 99)))
        best_j, threshold = -1.0, float(cands[0])
        for t in cands:
            j = (sig > t).mean() - (bkg > t).mean()
            if j > best_j:
                best_j, threshold = j, float(t)
        return threshold, best_j

    def _grid(self, frames, threshold):
        """Best (score, open, close, recall, precision) over the morphology
        grid at this threshold, scored against the configured objective."""
        match_iou = float(self._match_iou)
        opens = [int(x) for x in self._open_radii.split(",") if x.strip()]
        closes = [int(x) for x in self._close_radii.split(",") if x.strip()]
        best = (-1.0, opens[0], closes[0], 0.0, 0.0)

        for op in opens:
            for cl in closes:
                hit = tot = matched = emitted = 0
                for mot, boxes in frames:
                    got = boxes_from_mask(mot > threshold, op, cl,
                                          int(self._min_area),
                                          int(self._max_area),
                                          float(self._min_fill))
                    tot += len(boxes)
                    emitted += len(got)
                    for g in boxes:
                        if any(iou(g, d) >= match_iou for d in got):
                            hit += 1
                    for d in got:
                        if any(iou(g, d) >= match_iou for g in boxes):
                            matched += 1
                rec = hit / tot if tot else 0.0
                prec = matched / emitted if emitted else 0.0
                if self._objective == "f1":
                    score = (2 * rec * prec / (rec + prec)
                             if rec + prec > 0 else 0.0)
                else:
                    score = rec
                if score > best[0]:
                    best = (score, op, cl, rec, prec)
        return best

    def update_model(self):
        limit = int(self._max_samples)
        if self._objective not in ("recall", "f1"):
            print("Unknown objective '%s'; using recall." % self._objective)
            self._objective = "recall"

        # accepts a single value or a comma-separated list to sweep
        seps = [int(x) for x in str(self._frame_separation).split(",")
                if x.strip()]
        if not seps:
            print("No frame separation configured.")
            return

        overall = None
        for sep in seps:
            frames, sig, bkg = self._sample(sep, limit)
            if not frames or sig is None:
                print("  separation %d: not enough samples, skipped" % sep)
                continue
            threshold, best_j = self._threshold(sig, bkg)
            score, op, cl, rec, prec = self._grid(frames, threshold)
            print("  separation %d: threshold %.1f (J %.2f)  open %d close %d"
                  "  recall %.3f precision %.3f  %s %.3f"
                  % (sep, threshold, best_j, op, cl, rec, prec,
                     self._objective, score))
            if overall is None or score > overall[0]:
                overall = (score, sep, threshold, best_j, op, cl, rec, prec,
                           len(frames), len(sig))
            # one separation's motion images are freed before the next is
            # sampled, so peak memory does not grow with the sweep
            del frames

        if overall is None:
            print("No usable separation; cannot estimate settings.")
            return

        (_, sep, threshold, best_j, open_r, close_r, recall, precision,
         n_frames, n_ann) = overall
        print("Estimated from %d frames, %d annotations:" % (n_frames, n_ann))
        print("  frame separation %d" % sep)
        print("  threshold %.1f (Youden J %.2f)" % (threshold, best_j))
        print("  opening %d, closing %d (recall %.3f, precision %.3f "
              "at IoU %s)" % (open_r, close_r, recall, precision,
                              self._match_iou))

        # kernel_radius 0 is not a no-op: vxl_morphology builds a degenerate
        # structuring element and clears the mask. Below 1 is the identity.
        open_out = open_r if open_r > 0 else 0.5
        close_out = close_r if close_r > 0 else 0.5

        if self._output_directory and \
          not os.path.exists(self._output_directory):
            os.makedirs(self._output_directory)

        out = os.path.join(self._output_directory,
                           self._output_prefix + "_detector.pipe")
        with open(out, "w") as f:
            f.write(PIPELINE_TEMPLATE.format(
                frame_separation=sep,
                threshold=("%.1f" % threshold),
                open_radius=open_out,
                close_radius=close_out,
                min_area=self._min_area,
                max_area=self._max_area,
                min_fill=self._min_fill,
                class_name=self._class_name))
        print("Wrote %s" % out)


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        FrameDiffTrainer,
        "frame_diff",
        "Three-frame difference detector settings estimation",
    )
