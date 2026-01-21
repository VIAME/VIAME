# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import RefineDetections, RefineTracks
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObjectType
from kwiver.vital.util import VitalPIL
from kwiver.vital.types import ImageContainer
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import BoundingBoxD, ObjectTrackState, Track, ObjectTrackSet

from PIL import Image as PILImage

import numpy as np
import math
import delayed_image
from viame.pytorch.utilities import vital_config_update, vital_to_kwimage_box
from viame.pytorch.sam3_utilities import (
    mask_to_polygon, box_from_mask, image_to_rgb_numpy, get_autocast_context
)
from viame.core.segmentation_utils import (
    kwimage_mask_to_shapely,
    apply_polygon_policies,
    shapely_to_mask,
)

from distutils.util import strtobool


class Sam2Refiner(RefineDetections):
    """
    Full-Frame Classifier around Detection Sets

    CommandLine:
        xdoctest -m plugins/pytorch/sam2_refiner.py Sam2Refiner

    Example:
        >>> import torch
        >>> import kwimage
        >>> import ubelt as ub
        >>> print(torch.cuda.is_available())
        >>> sam2_checkpoint = ub.grabdata(
        >>>     'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        >>>     hash_prefix='0c4f89b91f1f951b95246f9544'
        >>> )
        >>> cfg_in = dict(
        >>>     sam2_checkpoint=sam2_checkpoint,
        >>>     sam2_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",  # needs to be relative to the modpath. Yuck.
        >>>     hole_policy="remove",
        >>>     multipolygon_policy='largest',
        >>>     #multipolygon_policy='convex_hull',
        >>>     #multipolygon_policy='allow',
        >>> )
        >>> # Create the refiner and init SAM2
        >>> self = Sam2Refiner()
        >>> self.set_configuration(cfg_in)
        >>> # Generate demo data to refine
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo()
        >>> image_id = 1
        >>> coco_image = dset.coco_image(image_id)
        >>> dets = coco_image.annots()[0:2].detections
        >>> imdata = coco_image.imdelay().finalize()
        >>> # Convert raw data structures to vital
        >>> image_data = vital_image_container_from_ndarray(imdata)
        >>> detections = kwimage_detections_to_vital(dets)
        >>> #
        >>> output = self.refine(image_data, detections)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> refined_dets = vital_detections_to_kwimage(output)
        >>> canvas1 = dets.draw_on(imdata.copy())
        >>> canvas2 = refined_dets.draw_on(imdata.copy())
        >>> canvas1 = kwimage.draw_header_text(canvas1, 'Input Image / Detections')
        >>> canvas2 = kwimage.draw_header_text(canvas2, 'Refined Detections')
        >>> canvas = kwimage.stack_images([canvas1, canvas2], axis=1, pad=10)
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
        >>> kwimage.imwrite('sam2_refined.jpg', canvas)  # For debugging

    Ignore:
        docker container cp amazing_boyd:/home/sam2_refined.jpg sam2_refined.jpg && eog sam2_refined.jpg
    """

    def __init__(self):
        RefineDetections.__init__(self)

        # kwiver configuration variables
        self._kwiver_config = {
            'cfg': "configs/sam2.1/sam2.1_hiera_b+.yaml",
            'checkpoint': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
            'device': "cuda",
            'overwrite_existing': 'True',
            'hole_policy': 'allow',  # can be allow or discard
            'multipolygon_policy': 'allow',  # can be allow, convex_hull, or largest
        }

        # netharn variables
        self._thresh = None
        self.predictor = None

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        hydra_overrides_extra = [
            # "++model.fill_hole_area=8",
        ]
        model = build_sam2(
            config_file=self._kwiver_config['cfg'],
            ckpt_path=self._kwiver_config['checkpoint'],
            device=self._kwiver_config['device'],
            mode='eval',
            hydra_overrides_extra=hydra_overrides_extra,
            apply_postprocessing=True,
        )
        self.predictor = SAM2ImagePredictor(model)
        self.overwrite_existing = strtobool(self._kwiver_config['overwrite_existing'])
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("cfg"):
            print("Requires a path to a config file (relative to the repo")
        if not cfg.has_value("checkpoint"):
            print("A checkpoint path to the weights needs to be specified!")
            return False
        return True

    def refine(self, image_data, detections) -> DetectedObjectSet:
        """
        Args:
            image_data (ImageContainer): image data corresponding to detections
            detections (DetectedObjectSet): input detections

        Returns:
            DetectedObjectSet: refined detections
        """
        import torch
        import kwimage
        import numpy as np

        if len(detections) == 0:
            return DetectedObjectSet()

        imdata = image_data.asarray().astype('uint8')

        # SAM2 requires RGB input
        if imdata.ndim == 2 or imdata.shape[2] == 1:
            imdata = np.repeat(imdata.reshape(imdata.shape[0], imdata.shape[1], 1), 3, axis=2)
        elif imdata.shape[2] == 4:
            imdata = imdata[:, :, :3]

        imdata = np.ascontiguousarray(imdata)
        predictor = self.predictor

        autocast_context = get_autocast_context(predictor.device)

        # Convert input vital types into kwimage
        kw_dets: kwimage.Detections = vital_detections_to_kwimage(detections)

        # TODO: can use more than boxes as prompts
        prompts = {}
        prompts['box'] = kw_dets.boxes.toformat('xyxy').data
        prompts['multimask_output'] = False

        with torch.inference_mode(), autocast_context:
            predictor.set_image(imdata)
            masks, scores, lowres_masks = predictor.predict(**prompts)
            masks = masks.astype(np.uint8)

        FIX_SQUEEZE_ISSUE = True
        if FIX_SQUEEZE_ISSUE:
            # Dont squeeze singleton dimensions, it breaks loop logic :'(
            if len(masks.shape) == 3:
                masks = masks[None, :, :, :]

        hole_policy = self._kwiver_config['hole_policy']
        mpoly_policy = self._kwiver_config['multipolygon_policy']

        # TODO: we may like to make these configrable
        # Area requires rasterio
        # pixels_are = 'areas'
        # origin_convention = 'corner'

        pixels_are = 'points'
        origin_convention = 'center'

        needs_polygon_postprocess = not (
            (hole_policy == 'allow') and
            (mpoly_policy == 'allow')
        )
        print(f'hole_policy={hole_policy}')
        print(f'mpoly_policy={mpoly_policy}')

        # Insert modified detections into a new DetectedObjectSet
        output = DetectedObjectSet()
        for vital_det, binmask in zip(detections, masks[:, 0, :, :]):

            # Extract the new mask info relative to the vital box
            box = vital_to_kwimage_box(vital_det.bounding_box)
            sl = box.quantize().to_slice()

            delayed = delayed_image.DelayedIdentity(binmask)
            relative_submask = delayed.crop(sl, clip=False, wrap=False).finalize()

            submask_dims: tuple = relative_submask.shape[0:2]
            relative_submask = relative_submask.reshape(submask_dims)

            if needs_polygon_postprocess:
                # Use shared utilities to convert mask -> shapely -> apply policies -> mask
                shape = kwimage_mask_to_shapely(
                    relative_submask,
                    pixels_are=pixels_are,
                    origin_convention=origin_convention,
                )

                if shape is not None:
                    shape = apply_polygon_policies(shape, hole_policy, mpoly_policy)

                    if shape is not None:
                        # Convert the polygon back to a mask
                        relative_submask = shapely_to_mask(
                            shape,
                            dims=submask_dims,
                            pixels_are=pixels_are,
                            origin_convention=origin_convention,
                        )

            # Modify detection and add to output list
            if vital_det.mask is None or self.overwrite_existing:
                vital_det.mask = vital_image_container_from_ndarray(relative_submask)
            output.add(vital_det)

        return output


class Sam2TrackRefiner(RefineTracks):
    """
    SAM2-based Track Refiner

    This refiner uses SAM2 for per-frame track refinement operations.
    It can improve mask quality and create new tracks from detections.

    Key features:
    - Re-segments existing track bounding boxes with SAM2 for better masks
    - Creates new tracks from single-state detections
    - Filters out tracks with low-quality masks
    - Adjusts bounding boxes to fit refined masks
    - Generates polygon outputs from masks

    Example:
        >>> from viame.pytorch.sam2_refiner import Sam2TrackRefiner
        >>> refiner = Sam2TrackRefiner()
        >>> refiner.set_configuration({'cfg': 'configs/sam2.1/sam2.1_hiera_b+.yaml'})
        >>> refined_tracks = refiner.refine(timestamp, image, tracks)
    """

    def __init__(self):
        RefineTracks.__init__(self)

        self._kwiver_config = {
            'cfg': "configs/sam2.1/sam2.1_hiera_b+.yaml",
            'checkpoint': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
            'device': "cuda",
            'overwrite_existing': 'True',
            'hole_policy': 'allow',
            'multipolygon_policy': 'allow',
            'min_mask_area': '10',
            'filter_by_quality': 'True',
            'adjust_boxes': 'True',
            'output_type': 'polygon',
            'polygon_simplification': '0.01',
        }

        self.predictor = None
        self._next_track_id = 1

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(RefineTracks, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        hydra_overrides_extra = []
        model = build_sam2(
            config_file=self._kwiver_config['cfg'],
            ckpt_path=self._kwiver_config['checkpoint'],
            device=self._kwiver_config['device'],
            mode='eval',
            hydra_overrides_extra=hydra_overrides_extra,
            apply_postprocessing=True,
        )
        self.predictor = SAM2ImagePredictor(model)

        # Parse config values
        self.overwrite_existing = strtobool(self._kwiver_config['overwrite_existing'])
        self._min_mask_area = int(self._kwiver_config['min_mask_area'])
        self._filter_by_quality = strtobool(self._kwiver_config['filter_by_quality'])
        self._adjust_boxes = strtobool(self._kwiver_config['adjust_boxes'])
        self._output_type = self._kwiver_config['output_type']
        self._polygon_simplification = float(self._kwiver_config['polygon_simplification'])
        self._hole_policy = self._kwiver_config['hole_policy']
        self._mpoly_policy = self._kwiver_config['multipolygon_policy']

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        if not cfg.has_value("cfg"):
            print("Requires a path to a config file (relative to the repo)")
            return False
        if not cfg.has_value("checkpoint"):
            print("A checkpoint path to the weights needs to be specified!")
            return False
        return True

    def refine(self, ts, image_data, tracks):
        """
        Refine tracks for the current frame.

        Args:
            ts: Timestamp for the current frame
            image_data: Image container for the current frame
            tracks: ObjectTrackSet containing tracks to refine

        Returns:
            ObjectTrackSet: Refined tracks
        """
        import torch

        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        frame_id = ts.get_frame()

        # Convert image to numpy RGB
        img_np = image_to_rgb_numpy(image_data)

        # Extract current frame's track states
        track_states = {}  # track_id -> (track, state, detection)
        max_track_id = 0

        for track in tracks.tracks():
            track_id = track.id()
            max_track_id = max(max_track_id, track_id)

            for state in track:
                if state.frame() == frame_id:
                    detection = state.detection()
                    track_states[track_id] = (track, state, detection)
                    break

        # Update next track ID to be higher than any existing
        self._next_track_id = max(self._next_track_id, max_track_id + 1)

        # Collect boxes for segmentation
        boxes_to_segment = []
        box_sources = []  # track_id

        for tid, (track, state, det) in track_states.items():
            bbox = det.bounding_box
            box = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
            boxes_to_segment.append(box)
            box_sources.append(tid)

        # Segment all boxes with SAM2
        if len(boxes_to_segment) > 0:
            masks = self._segment_with_sam2(img_np, boxes_to_segment)
        else:
            masks = []

        # Process results and build output tracks
        output_tracks = []
        processed_track_ids = set()

        for i, (mask, tid) in enumerate(zip(masks, box_sources)):
            mask_area = np.sum(mask)

            # Filter by minimum mask area
            if self._filter_by_quality and mask_area < self._min_mask_area:
                processed_track_ids.add(tid)
                continue

            track, old_state, old_det = track_states[tid]
            processed_track_ids.add(tid)

            # Create refined detection
            new_det = self._create_refined_detection(old_det, mask)

            # Create new track state
            new_state = ObjectTrackState(ts, new_det.bounding_box,
                                        new_det.confidence, new_det)

            # Rebuild track with new state for this frame
            new_history = []
            for state in track:
                if state.frame() == frame_id:
                    new_history.append(new_state)
                else:
                    new_history.append(state)

            new_track = Track(tid, new_history)
            output_tracks.append(new_track)

        # Include tracks that have no state for current frame (preserve history)
        for track in tracks.tracks():
            tid = track.id()
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(track)

        return ObjectTrackSet(output_tracks)

    def _segment_with_sam2(self, image_np, boxes):
        """
        Segment objects in image using SAM2 with box prompts.

        Args:
            image_np: RGB image as numpy array
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks (numpy arrays)
        """
        import torch

        if len(boxes) == 0:
            return []

        autocast_context = get_autocast_context(self.predictor.device)

        prompts = {
            'box': np.array(boxes),
            'multimask_output': False
        }

        with torch.inference_mode(), autocast_context:
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict(**prompts)

        # Handle shape - ensure we have [N, 1, H, W] or similar
        if len(masks.shape) == 3:
            masks = masks[None, :, :, :]

        return [masks[i, 0] for i in range(len(boxes))]

    def _create_refined_detection(self, old_det, mask):
        """
        Create a refined detection from an existing detection and new mask.

        Args:
            old_det: Original DetectedObject
            mask: New binary mask from SAM2

        Returns:
            DetectedObject: Refined detection
        """
        # Get bounding box
        if self._adjust_boxes:
            bbox = box_from_mask(mask)
            if bbox is None:
                bbox = old_det.bounding_box
        else:
            bbox = old_det.bounding_box

        # Copy classification
        det_type = old_det.type
        confidence = old_det.confidence

        # Create new detection
        new_det = DetectedObject(bbox, confidence, det_type)

        # Add polygon
        if self._output_type in ('polygon', 'both'):
            polygon = mask_to_polygon(mask, self._polygon_simplification)
            if polygon is not None:
                new_det.set_polygon(polygon)

        # Add mask (relative to bounding box)
        if self.overwrite_existing or old_det.mask is None:
            # Create relative mask within bounding box
            x1, y1 = int(bbox.min_x()), int(bbox.min_y())
            x2, y2 = int(bbox.max_x()), int(bbox.max_y())

            # Ensure bounds are within mask dimensions
            h, w = mask.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            relative_mask = mask[y1:y2, x1:x2].astype(np.uint8)
            if relative_mask.size > 0:
                new_det.mask = vital_image_container_from_ndarray(relative_mask)

        return new_det


def kwimage_boxes_to_vital(boxes):
    """
    Converts kwimage box data structures to vital.

    Args:
        boxes (kwimage.Boxes): input kwimage boxes

    Returns:
        List[BoundingBoxI | BoundingBoxF | BoundingBoxD]:
            The vital bbox objects

    Example:
        >>> # xdoctest: +REQUIRES(module:kwiver.vital)
        >>> boxes = kwimage.Boxes.random(10)
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(np.int8))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(np.int32))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(np.int64))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(np.float32))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(np.float64))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(float))
        >>> kwimage_boxes_to_vital(boxes.scale(100).astype(int))
    """
    from kwiver.vital.types import BoundingBoxI
    # from kwiver.vital.types import BoundingBoxF
    from kwiver.vital.types import BoundingBoxD

    # Determine which bbox type is needed
    if boxes.data.dtype.kind == 'f':
        # not sure if using BoundingBoxF makes sense or works with other classes
        # if boxes.data.dtype.itemsize == 4:
        #     box_cls = BoundingBoxF
        # elif boxes.data.dtype.itemsize == 8:
        box_cls = BoundingBoxD
        # else:
        #     raise TypeError(boxes.data.dtype)
    elif boxes.data.dtype.kind == 'i':
        box_cls = BoundingBoxI
    else:
        raise TypeError(boxes.data.dtype)

    # Convert to the bbox format vital likes and make the objects
    xyxy_data = boxes.to_ltrb().data
    vital_boxes = [
        box_cls(xmin, ymin, xmax, ymax)
        for xmin, ymin, xmax, ymax in xyxy_data]
    return vital_boxes


def kwimage_detections_to_vital(kwimage_dets):
    """
    Convert kwimage Detection data structures to vital.

    Args:
        kwimage_dets (kwimage.Detections):

    Returns:
        DetectedObjectSet: converted detections

    Example:
        >>> # xdoctest: +REQUIRES(module:kwiver.vital)
        >>> # Test with everything
        >>> dets = kwimage.Detections.random(10, segmentations=True).scale(64)
        >>> vital_dets = kwimage_detections_to_vital(dets)
        >>> # Test without segmentations
        >>> dets.data.pop('segmentations')
        >>> vital_dets = kwimage_detections_to_vital(dets)
        >>> # Test without classes
        >>> dets.meta.pop('classes')
        >>> dets.data.pop('class_idxs')
        >>> vital_dets = kwimage_detections_to_vital(dets)
        >>> # Test without scores
        >>> dets.data.pop('scores')
        >>> vital_dets = kwimage_detections_to_vital(dets)
    """

    dets = kwimage_dets

    boxes = dets.boxes  # required
    scores = dets.data.get('scores', None)
    class_idxs = dets.data.get('class_idxs', None)
    segmentations = dets.data.get('segmentations', None)
    classes = dets.meta.get('classes', None)

    if scores is None:
        scores = np.ones(len(dets))
    if class_idxs is None:
        class_idxs = [None] * len(dets)
    if segmentations is None:
        segmentations = [None] * len(dets)

    vital_boxes = kwimage_boxes_to_vital(boxes)

    vital_dets = []
    for bbox, score, class_idx, sseg in zip(vital_boxes, scores, class_idxs, segmentations):
        detobjkw = {
            'bbox': bbox,
        }
        if score is not None:
            detobjkw['confidence'] = score

        if sseg is not None:
            # Convert the segmentation to vital
            # Convert whatever type of kwimage segmenation it is into a mask
            # This might be incorrect... is there a way to determine
            # if the masks are relative to the bbox or relative to the image?
            # for float bboxes the second doesn't make much sense, but the
            # first takes much more data.
            # TODO: in kwimage to denote which space masks belong to.
            if hasattr(sseg, 'to_relative_mask'):
                # sseg is a Polygon, and has this method
                offset = [int(math.ceil(bbox.min_x())), int(math.ceil(bbox.min_y()))]
                bottom = [int(math.floor(bbox.max_x())), int(math.floor(bbox.max_y()))]
                dsize = np.array(bottom) - np.array(offset)
                dims = tuple(map(int, dsize[::-1]))
                mask = kwimage_new_to_relative_mask(sseg, offset=offset, dims=dims)
                # If mask is not relative use:
                # x1, y1, x2, y2 = sseg.to_boxes().to_ltrb().data[0]
                # image_dims = (int(math.ceil(y2)), int(math.ceil(x2)))
                # mask = sseg.to_mask(dims=image_dims)
            else:
                # sseg is probably a mask already
                # TODO: handle relativeness
                mask = sseg.data.to_c_mask()
                print('case2')
                raise NotImplementedError('implement correctly if needed')

            nd_mask = mask.to_c_mask().data
            vital_mask = vital_image_container_from_ndarray(nd_mask)
            detobjkw['mask'] = vital_mask

        if classes is not None and class_idx is not None:
            label = classes[class_idx]
            detobjkw['classifications'] = DetectedObjectType(label, score)

        obj = DetectedObject(**detobjkw)
        vital_dets.append(obj)

    vital_dets = DetectedObjectSet(vital_dets)
    return vital_dets


def kwimage_new_to_relative_mask(self, offset=None, dims=None, return_offset=False):
    """
    Returns a translated mask such the mask dimensions are minimal.

    In other words, we move the polygon all the way to the top-left and
    return a mask just big enough to fit the polygon.

    Args:
        offset (None):
            if specified, return the mask relative to this xy location.
            If unspecified, it uses the corner of the segmentation.

        dims (None):
            the h, w of the new mask, relative to the offset

    Returns:
        kwimage.Mask
    """
    x, y, w, h = self.to_boxes().quantize().to_xywh().data[0]
    if offset is None:
        offset = (x, y)

    if dims is None:
        dims = (
            x - offset[0] + w,
            x - offset[1] + h
        )
    translation = tuple(-p for p in offset)
    mask = self.translate(translation).to_mask(dims=dims)
    if return_offset:
        offset = (x, y)
        return mask, offset
    else:
        return mask


def vital_detections_to_kwimage(vital_dets):
    """
    Convert vital detection objects into kwimage

    Args:
        detections (List[kwiver.vital.types.DetectedObject] | DetectedObjectSet)

    Returns:
        kwimage.Detections

    Example:
        >>> # xdoctest: +REQUIRES(module:kwiver.vital)
        >>> kwimage_dets = kwimage.Detections.random(10, segmentations=True).scale(256)
        >>> vital_dets = kwimage_detections_to_vital(kwimage_dets)
        >>> # Do some round trips
        >>> recon_kwimage_dets1 = vital_detections_to_kwimage(vital_dets)
        >>> recon_vital_dets1 = kwimage_detections_to_vital(recon_kwimage_dets1)
        >>> recon_kwimage_dets2 = vital_detections_to_kwimage(recon_vital_dets1)
        >>> # should be the same
        >>> print([p.area for p in kwimage_dets.data['segmentations']])
        >>> print([p.to_multi_polygon().area for p in recon_kwimage_dets1.data['segmentations']])
        >>> print([p.to_multi_polygon().area for p in recon_kwimage_dets2.data['segmentations']])
        >>> for i in range(10):
        >>>     print(recon_kwimage_dets1.data['segmentations'][i].data.to_c_mask().data.sum())
        >>>     print(recon_kwimage_dets2.data['segmentations'][i].data.to_c_mask().data.sum())
    """
    import kwimage
    boxes = []
    scores = []
    masks = []
    catnames = []

    detkw = {}

    all_catnames = set()
    for obj in vital_dets:
        bbox = obj.bounding_box
        xyxy = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
        offset = xyxy[0:2]
        boxes.append(xyxy)
        scores.append(obj.confidence)
        if obj.mask is not None and obj.mask.width() > 0 and obj.mask.height() > 0:
            vital_mask = obj.mask.image()
            nd_mask = vital_mask.asarray()
            if len(nd_mask.shape) == 3:
                assert nd_mask.shape[2] == 1
                nd_mask = nd_mask[:, :, 0]
            # In vital, we assume the mask is relative to the top left of the box
            # Compute the offset and "effective" shape for kwimage.
            x, y = offset
            h, w = nd_mask.shape[0:2]
            h += y
            w += x
            shape = (int(h), int(w))
            kw_mask = kwimage.Mask.from_mask(nd_mask, offset=offset, shape=shape)
        else:
            kw_mask = None

        if 0:
            # is this necessary?
            all_catnames.update(set(obj.type.all_class_names()))

        catname = obj.type.get_most_likely_class()
        all_catnames.add(catname)
        catnames.append(catname)
        # obj.type.get_most_likely_score()  # not handled
        masks.append(kw_mask)

    if len(masks):
        detkw['segmentations'] = kwimage.SegmentationList(masks)

    classes = sorted(all_catnames)
    if len(classes):
        catname_to_idx = {name: idx for idx, name in enumerate(classes)}
        class_idxs = np.array([catname_to_idx[cname] for cname in catnames])
        detkw['class_idxs'] = class_idxs
        detkw['classes'] = classes

    detkw['boxes'] = kwimage.Boxes(np.array(boxes), format='ltrb')
    detkw['scores'] = np.array(scores)

    dets = kwimage.Detections(**detkw)
    return dets


def vital_image_container_from_ndarray(ndarray_img):
    """
    Args:
        ndarray_img (np.ndarray): input image as an ndarray

    Returns:
        kwiver.vital.types.ImageContainer
    """
    pil_img = PILImage.fromarray(ndarray_img)
    vital_img = ImageContainer(VitalPIL.from_pil(pil_img))
    return vital_img


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Sam2Refiner (RefineDetections)
    implementation_name = "sam2"

    if not algorithm_factory.has_algorithm_impl_name(
            Sam2Refiner.static_type_name(), implementation_name):
        algorithm_factory.add_algorithm(
            implementation_name, "SAM2-based detection refiner",
            Sam2Refiner)

        algorithm_factory.mark_algorithm_as_loaded(implementation_name)

    # Register Sam2TrackRefiner (RefineTracks)
    track_impl_name = "sam2"

    if not algorithm_factory.has_algorithm_impl_name(
            Sam2TrackRefiner.static_type_name(), track_impl_name):
        algorithm_factory.add_algorithm(
            track_impl_name, "SAM2-based track refiner for adding segmentation masks to tracks",
            Sam2TrackRefiner)

        algorithm_factory.mark_algorithm_as_loaded(track_impl_name)
