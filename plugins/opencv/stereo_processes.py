# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# -*- coding: utf-8 -*-
"""
CommandLine:
    cd ~/code/VIAME/plugins/opencv/python
    export PYTHONPATH=$(pwd):$PYTHONPATH

    cd ~/code/VIAME/build
    export KWIVER_PLUGIN_PATH=""
    export SPROKIT_MODULE_PATH=""
    source install/setup_viame.sh

    export KWIVER_DEFAULT_LOG_LEVEL=info
    export KWIVER_PYTHON_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes

    python ~/code/VIAME/plugins/opencv/python/run_opencv.py

    ~/code/VIAME/build/install/bin/viame ~/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe
    ~/code/VIAME/build/install/bin/viame opencv.pipe -S pythread_per_process

    ~/code/VIAME/build/install/bin/viame ~/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe  -S pythread_per_process
    ~/code/VIAME/build/install/bin/viame opencv.pipe -S pythread_per_process

SeeAlso
    ~/code/VIAME/packages/kwiver/vital/bindings/python/vital/types
"""

import numpy as np

from kwiver.vital.types import (
    BoundingBoxD,
    DetectedObject,
    DetectedObjectSet,
    DetectedObjectType,
    ImageContainer,
    ObjectTrackState,
    ObjectTrackSet,
    Point2d,
    Track,
)

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

import ubelt as ub
import os
import itertools as it
import cv2

from . import stereo_algos as ctalgo

from kwiver.vital import vital_logging

logger = vital_logging.getLogger(__name__)
print = logger.info


# TODO: add something similar in sprokit proper
TMP_SPROKIT_PROCESS_REGISTRY = []


def tmp_sprokit_register_process(name=None, doc=''):
    def _wrp(cls):
        name_ = name
        if name is None:
            name_ = cls.__name__
        TMP_SPROKIT_PROCESS_REGISTRY.append((name_, doc, cls))
        return cls
    return _wrp


def opencv_setup_config(self, default_params):
    if isinstance(default_params, dict):
        default_params = list(it.chain(*default_params.values()))
    for pi in default_params:
        self.add_config_trait(pi.name, pi.name, str(pi.default), pi.doc)
        self.declare_config_using_trait(pi.name)


def tmp_smart_cast_config(self):
    # import ubelt as ub
    # import utool as ut
    config = {}
    keys = [k for k in list(self.available_config())
            if not k.startswith('_')]
    for key in keys:
        strval = self.config_value(key)
        # print('strval = {!r}'.format(strval))
        try:
            val = eval(strval, {}, {})
        except Exception:
            val = strval
        config[key] = val
    return config


@tmp_sprokit_register_process(name='gmm_motion_detector',
                              doc='preliminatry fish detection')
class GMMDetectFishProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.

    """
    # --------------------------------------------------------------------------
    def __init__(self, conf):
        print('conf = {!r}'.format(conf))
        logger.debug(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        opencv_setup_config(self, ctalgo.GMMForegroundObjectDetector.default_params())

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        self.declare_output_port_using_trait('detected_object_set', optional )

    # --------------------------------------------------------------------------
    def _configure(self):
        logger.debug(' ----- configure ' + self.__class__.__name__)
        config = tmp_smart_cast_config(self)
        print('detector config = {}'.format(ub.repr2(config, nl=2)))
        self.detector = ctalgo.GMMForegroundObjectDetector(**config)
        self._base_configure()

    # --------------------------------------------------------------------------
    def _dowork(self, img_container):
        """
        Helper to decouple the algorithm and pipeline logic

        CommandLine:
            xdoctest viame.processes.opencv.processes GMMDetectFishProcess._dowork

        Example:
            >>> from viame.processes.opencv.processes import *
            >>> from kwiver.vital.types import ImageContainer
            >>> import kwiver.sprokit.pipeline.config
            >>> # construct dummy process instance
            >>> conf = kwiver.sprokit.pipeline.config.empty_config()
            >>> self = GMMDetectFishProcess(conf)
            >>> self._configure()
            >>> # construct test data
            >>> from vital.util import VitalPIL
            >>> from PIL import Image as PILImage
            >>> pil_img = PILImage.open(ub.grabdata('https://i.imgur.com/Jno2da3.png'))
            >>> pil_img = PILImage.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            >>> img_container = ImageContainer(VitalPIL.from_pil(pil_img))
            >>> # Initialize the background detector by sending 10 black frames
            >>> for i in range(10):
            >>>     empty_set = self._dowork(img_container)
            >>> # now add a white box that should be detected
            >>> np_img = np.zeros((512, 512, 3), dtype=np.uint8)
            >>> np_img[300:340, 220:380] = 255
            >>> img_container = ImageContainer.fromarray(np_img)
            >>> detection_set = self._dowork(img_container)
            >>> assert len(detection_set) == 1
            >>> obj = detection_set[0]
        """
        # This should be read as np.uint8
        np_img = img_container.asarray()

        detection_set = DetectedObjectSet()
        ct_detections = self.detector.detect(np_img)

        for detection in ct_detections:
            bbox = BoundingBoxD(*detection.bbox.coords)
            mask = detection.mask.astype(np.uint8)
            vital_mask = ImageContainer.fromarray(mask)
            dot = DetectedObjectType("Motion", 1.0)
            obj = DetectedObject(bbox, 1.0, dot, mask=vital_mask)
            detection_set.add(obj)
        return detection_set

    def _step(self):
        logger.debug(' ----- ' + self.__class__.__name__ + ' step')
        # grab image container from port using traits
        img_container = self.grab_input_using_trait('image')

        # Process image container
        detection_set = self._dowork(img_container)

        # Push the output
        self.push_to_port_using_trait('detected_object_set', detection_set)

        self._base_step()


# Not registered: the C++ measure_objects_process in this same plugin claims
# measure_using_stereo, and being a compiled plugin it always registered first,
# so this prototype was unreachable even before kwiver made a duplicate plugin
# name a hard error rather than a warning.
class MeasureProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.
    """
    # --------------------------------------------------------------------------
    def __init__(self, conf):
        logger.debug(' ----- ' + self.__class__.__name__ + ' init')

        KwiverProcess.__init__(self, conf)

        opencv_setup_config(self, ctalgo.StereoLengthMeasurments.default_params())

        self.add_config_trait('measurement_file', 'measurement_file', '',
                              'output file to write detection measurements')
        self.declare_config_using_trait('measurement_file')
        self.add_config_trait('calibration_file', 'calibration_file', 'cal_201608.mat',
                              'matlab or npz file with calibration info')
        self.declare_config_using_trait('calibration_file')

        optional = process.PortFlags()

        required = process.PortFlags()
        required.add(self.flag_required)

        self.add_port_trait('detected_object_set1',
          'detected_object_set',
          'Detections from camera1')
        self.add_port_trait('detected_object_set2',
          'detected_object_set',
          'Detections from camera2')

        self.add_port_trait('object_track_set1',
          'object_track_set',
          'Output tracks for camera1')
        self.add_port_trait('object_track_set2',
          'object_track_set',
          'Output tracks for camera2')

        #  declare our input ports ( port-name,flags )
        self.declare_input_port_using_trait('detected_object_set1', required)
        self.declare_input_port_using_trait('detected_object_set2', required)

        self.declare_input_port_using_trait('timestamp', required)

        #  declare our output ports ( port-name,flags )
        self.declare_output_port_using_trait('detected_object_set1', optional)
        self.declare_output_port_using_trait('detected_object_set2', optional)

        self.declare_output_port_using_trait('object_track_set1', required)
        self.declare_output_port_using_trait('object_track_set2', required)

    # --------------------------------------------------------------------------
    def _configure(self):
        logger.debug(' ----- ' + self.__class__.__name__ + ' configure')
        config = tmp_smart_cast_config(self)

        logger.info('triangulator config = {}'.format(ub.repr2(config, nl=2)))
        self.measurement_file = config.pop('measurement_file')
        self.calibration_file = config.pop('calibration_file')
        self.triangulator = ctalgo.StereoLengthMeasurments(**config)

        # Load camera calibration data here.
        if not os.path.exists(self.calibration_file):
            raise KeyError('must specify a valid camera calibration path')

        self.cal = ctalgo.StereoCalibration.from_file(self.calibration_file)
        logger.info('self.cal = {!r}'.format(self.cal))

        self.headers = ['current_frame', 'fishlen', 'range', 'error', 'dz',
                        'box_pts1', 'box_pts2']

        if self.measurement_file:
            self.output_file = open(self.measurement_file, 'w')
            self.output_file.write(','.join(self.headers) + '\n')
            self.output_file.close()

            self.output_file = open(self.measurement_file, 'a')

        self._base_configure()

        self.prog = ub.ProgIter(verbose=3)
        self.prog.begin()

        self.frame_id = 0
        self.track_id = 0

    # --------------------------------------------------------------------------
    def _step(self):
        logger.debug(' ----- ' + self.__class__.__name__ + ' step')
        self.prog.step()

        if self.cal is None:
            self.cal = True
            logger.debug(' ----- ' + self.__class__.__name__ + ' grab cam1')
            # grab camera only if we dont have one yet
            camera1 = self.grab_input_using_trait('camera' + '1')
            logger.debug(' ----- ' + self.__class__.__name__ + ' grab cam2')
            camera2 = self.grab_input_using_trait('camera' + '2')

            def _cal_from_vital(vital_camera):
                vci = vital_camera.intrinsics
                cam_dict = {
                    'extrinsic': {
                        'om': vital_camera.rotation.rodrigues().ravel(),
                        'T': vital_camera.translation.ravel(),
                    },
                    'intrinsic': {
                        'cc': vci.principle_point.ravel(),
                        'fc': [vci.focal_length,
                               vci.focal_length / vci.aspect_ratio],
                        'alpha_c': vci.skew,
                        'kc': vci.dist_coeffs.ravel(),
                    }
                }
                return cam_dict

            logger.debug(' ----- ' + self.__class__.__name__ + ' parse cameras')
            self.cal = ctalgo.StereoCalibration({
                'left': _cal_from_vital(camera1),
                'right': _cal_from_vital(camera2),
            })
            logger.debug(' ----- ' + self.__class__.__name__ + ' no more need for cameras')

        detection_set1 = self.grab_input_using_trait('detected_object_set' + '1')
        detection_set2 = self.grab_input_using_trait('detected_object_set' + '2')

        timestamp = self.grab_input_using_trait('timestamp')

        # Convert back to the format the algorithm understands
        def _detections_from_vital(detection_set):
            for vital_det in detection_set:
                bbox = vital_det.bounding_box
                coords = [bbox.min_x(), bbox.min_y(),
                          bbox.max_x(), bbox.max_y()]
                if vital_det.mask:
                    mask = vital_det.mask.asarray()
                else:
                    mask = None
                ct_bbox = ctalgo.BoundingBox(coords)

                # Measure between the detector's own head/tail points when it
                # emits them; otherwise the box proxies are used downstream.
                kps = vital_det.keypoints
                special_keypoints = None
                if kps and 'head' in kps and 'tail' in kps:
                    special_keypoints = {
                        name: np.asarray(kps[name].value, dtype=float)
                        for name in ('head', 'tail')
                    }

                ct_det = ctalgo.DetectedObject(
                    ct_bbox, mask, special_keypoints=special_keypoints)
                yield ct_det

        detections1 = list(_detections_from_vital(detection_set1))
        detections2 = list(_detections_from_vital(detection_set2))

        assignment, assign_data, cand_errors = self.triangulator.find_matches(
            self.cal, detections1, detections2)

        logger.debug(' ----- ' + self.__class__.__name__ + ' found {} matches'.format(len(assign_data)))

        # Append assignments to the measurements
        if self.measurement_file:
            def csv_repr(d):
                if isinstance(d, np.ndarray):
                    d = d.tolist()
                s = repr(d)
                return s.replace('\n', '').replace(',', ';').replace(' ', '')

            for data in assign_data:
                data['current_frame'] = self.frame_id
                self.frame_id = self.frame_id + 1
                line = ','.join([csv_repr(d) for d in ub.take(data, self.headers)])
                self.output_file.write(line + '\n')

            if assign_data:
                 self.output_file.flush()

        # Create output detection vectors
        output_dets1 = [d for d in detection_set1]
        output_dets2 = [d for d in detection_set2]

        output_trks1 = []
        output_trks2 = []

        has_match1 = [False] * len(detection_set1)
        has_match2 = [False] * len(detection_set2)

        # Assign all lengths to detections and generate matched tracks
        for match in assign_data:
            i1 = match["ij"][0]
            i2 = match["ij"][1]

            has_match1[i1] = True
            has_match2[i2] = True

            output_dets1[i1].set_attribute("length", match["fishlen"])
            output_dets2[i2].set_attribute("length", match["fishlen"])

            head, tail = detections1[i1].center_keypoints()
            output_dets1[i1].add_keypoint('head', Point2d(float(head[0]), float(head[1])))
            output_dets1[i1].add_keypoint('tail', Point2d(float(tail[0]), float(tail[1])))
            head, tail = detections2[i2].center_keypoints()
            output_dets2[i2].add_keypoint('head', Point2d(float(head[0]), float(head[1])))
            output_dets2[i2].add_keypoint('tail', Point2d(float(tail[0]), float(tail[1])))

            state1 = ObjectTrackState(timestamp, output_dets1[i1])
            state2 = ObjectTrackState(timestamp, output_dets2[i2])
            track1 = Track(id=self.track_id)
            track2 = Track(id=self.track_id)
            track1.append(state1)
            track2.append(state2)
            output_trks1.append(track1)
            output_trks2.append(track2)
            self.track_id = self.track_id + 1

        # Add unmatched detections to tracks lists
        for detection, matched in zip(output_dets1, has_match1):
            if not matched:
                state = ObjectTrackState(timestamp, detection)
                track = Track(id=self.track_id)
                track.append(state)
                output_trks1.append(track)
                self.track_id = self.track_id + 1
        for detection, matched in zip(output_dets2, has_match2):
            if not matched:
                state = ObjectTrackState(timestamp, detection)
                track = Track(id=self.track_id)
                track.append(state)
                output_trks2.append(track)
                self.track_id = self.track_id + 1

        # Format outputs as vital sets
        detection_set1 = DetectedObjectSet(output_dets1)
        detection_set2 = DetectedObjectSet(output_dets2)

        self.push_to_port_using_trait('detected_object_set1', detection_set1)
        self.push_to_port_using_trait('detected_object_set2', detection_set2)

        track_set1 = ObjectTrackSet(output_trks1)
        track_set2 = ObjectTrackSet(output_trks2)

        self.push_to_port_using_trait('object_track_set1', track_set1)
        self.push_to_port_using_trait('object_track_set2', track_set2)
        self._base_step()

@tmp_sprokit_register_process(name='unrectified_keypoint_stereo_measure',
                              doc='Measure length using head/tail keypoints + stereoscopic image pairs')
class UnrectifiedKeypointStereoMeasureProcess(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)
        self.add_config_trait('calibration_file', 'calibration_file', '', 'Calibration file')
        self.add_config_trait('search_radius', 'search_radius', '15', 'Circle radius (pixels) to search for around keypoints')
        self.add_config_trait('disparity_percentile', 'disparity_percentile', '95', 'Disparity percentile')
        self.add_config_trait('matching_tolerance', 'matching_tolerance', '60.0', 'Max verticale tolerance for matching')
        self.add_config_trait('num_disparities', 'num_disparities', '240', 'Maximum disparity minus minimum disparity. Must be divisible by 16.')
        self.add_config_trait('block_size', 'block_size', '11', 'Block size for SGBM algorithm')
        self.add_config_trait('samples', 'samples', '11', 'Number of depth samples along head/tail axis')
        self.add_config_trait('max_outliers', 'max_outliers', '3', 'Maximum number of outliers to compute length')

        self.declare_config_using_trait('calibration_file')
        self.declare_config_using_trait('search_radius')
        self.declare_config_using_trait('disparity_percentile')
        self.declare_config_using_trait('matching_tolerance')
        self.declare_config_using_trait('num_disparities')
        self.declare_config_using_trait('block_size')
        self.declare_config_using_trait('samples')
        self.declare_config_using_trait('max_outliers')

        required = process.PortFlags()
        required.add(self.flag_required)

        self.add_port_trait('image1', 'image', 'Image from camera1')
        self.add_port_trait('image2', 'image', 'Image from camera2')
        self.declare_input_port_using_trait('image1', required)
        self.declare_input_port_using_trait('image2', required)

        self.add_port_trait('object_track_set1', 'object_track_set', 'Tracks from camera1')
        self.add_port_trait('object_track_set2', 'object_track_set', 'Tracks from camera2')
        self.add_port_trait('timestamp', 'timestamp', 'timestamp')

        self.add_port_trait('object_track_set_out1', 'object_track_set', 'Output tracks for camera1')
        self.add_port_trait('object_track_set_out2', 'object_track_set', 'Output tracks for camera2')

        self.declare_input_port_using_trait('object_track_set1', required)
        self.declare_input_port_using_trait('object_track_set2', required)
        self.declare_input_port_using_trait('timestamp', required)

        self.declare_output_port_using_trait('object_track_set_out1', process.PortFlags())
        self.declare_output_port_using_trait('object_track_set_out2', process.PortFlags())

        self.Q = None

        self.right_track_memory = set()
        self.locked_matches = {}
        self.ID_OFFSET = 1000000

    def _configure(self):
        cal_path = self.config_value('calibration_file')
        if not cal_path:
            raise ValueError("calibration_file is required.")

        self.search_radius = int(self.config_value('search_radius'))
        self.disparity_percentile = float(self.config_value('disparity_percentile'))
        self.matching_tolerance = float(self.config_value('matching_tolerance'))
        self.num_disp = int(self.config_value('num_disparities'))
        self.block_size = int(self.config_value('block_size'))
        self.samples = int(self.config_value('samples'))
        self.max_outliers = int(self.config_value('max_outliers'))

        self.cal = ctalgo.StereoCalibration.from_file(cal_path)

        self.K1, self.K2 = self.cal.intrinsic_matrices()
        self.D1 = self.cal.data['left']['intrinsic']['kc']
        self.D2 = self.cal.data['right']['intrinsic']['kc']

        om_right = self.cal.data['right']['extrinsic']['om']
        T_right = self.cal.data['right']['extrinsic']['T']

        self.R, _ = cv2.Rodrigues(om_right)
        self.T = T_right.reshape(3, 1)

        Tx = np.array([
            [0, -self.T[2,0], self.T[1,0]],
            [self.T[2,0], 0, -self.T[0,0]],
            [-self.T[1,0], self.T[0,0], 0]
        ])
        E = Tx @ self.R

        K2_inv_T = np.linalg.inv(self.K2).T
        K1_inv = np.linalg.inv(self.K1)
        self.F = K2_inv_T @ E @ K1_inv

        self._base_configure()

    def extract_head_tail(self, detection: DetectedObject):
        kpts = detection.keypoints
        if not kpts:
            return None, None

        head_pt = kpts.get('head', None)
        tail_pt = kpts.get('tail', None)
        if head_pt is None or tail_pt is None:
            return None, None

        return np.array([head_pt.value[0], head_pt.value[1]]), np.array([tail_pt.value[0], tail_pt.value[1]])

    def rectify_point(self, pt_unrectified, K, D, R, P):
        pt_cv = np.array([[[pt_unrectified[0], pt_unrectified[1]]]], dtype=np.float64)
        pt_rect = cv2.undistortPoints(pt_cv, K, D, R=R, P=P)
        return pt_rect[0, 0]

    def compute_epipolar_distance(self, pt1, pt2):
        p1 = np.array([pt1[0], pt1[1], 1.0])
        l2 = self.F @ p1
        a, b, c = l2[0], l2[1], l2[2]
        dist = abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)
        return dist


    def get_sample_3d(self, sample_pt_rect, disp_map):
        sx, sy = int(round(sample_pt_rect[0])), int(round(sample_pt_rect[1]))
        h, w = disp_map.shape

        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            return None, None

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (sx, sy), radius=self.search_radius, color=255, thickness=-1)

        circle_disp = disp_map[mask == 255]
        valid_disp = circle_disp[circle_disp > 0.5]

        if len(valid_disp) < (0.50 * len(circle_disp)):
            return None, None

        best_disp = np.percentile(valid_disp, self.disparity_percentile)
        if best_disp < 1.0:
            return None, None

        pt3d_hom = self.Q @ np.array([sx, sy, best_disp, 1.0])
        pt3d = pt3d_hom[:3] / pt3d_hom[3]

        return pt3d.flatten(), best_disp

    def _step(self):
        img_c1 = self.grab_input_using_trait('image1')
        img_c2 = self.grab_input_using_trait('image2')
        img1 = img_c1.asarray()
        img2 = img_c2.asarray()

        if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        track_set1: ObjectTrackSet = self.grab_input_using_trait('object_track_set1')
        track_set2: ObjectTrackSet = self.grab_input_using_trait('object_track_set2')
        out_track_set1 = [d for d in track_set1.tracks()]
        out_track_set2 = [d for d in track_set2.tracks()]

        if self.Q is None:
            h, w = img1.shape[:2]
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
                self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T
            )
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1, (w, h), cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2, (w, h), cv2.CV_32FC1)

            self.stereo_left = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disp, blockSize=self.block_size,
                P1=8 * self.block_size**2, P2=32 * self.block_size**2,
                disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
            )
            self.stereo_right = cv2.ximgproc.createRightMatcher(self.stereo_left)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_left)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.5)
            self.white_mask = np.ones(img1.shape[:2], dtype=np.uint8) * 255

        rect1 = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)

        valid_roi_mask = cv2.remap(self.white_mask, self.map1x, self.map1y, cv2.INTER_LINEAR)

        disp_left_raw = self.stereo_left.compute(rect1, rect2)
        disp_right_raw = self.stereo_right.compute(rect2, rect1)

        disp_left_filtered = self.wls_filter.filter(disp_left_raw, rect1, None, disp_right_raw)

        # Double strict mask: reject SGBM failures and artificial black borders
        invalid_sgbm_mask = disp_left_raw <= 0
        invalid_border_mask = valid_roi_mask < 200

        disp_left_filtered[invalid_sgbm_mask] = -16
        disp_left_filtered[invalid_border_mask] = -16

        disp_left = disp_left_filtered.astype(np.float32) / 16.0

        ts_datum = self.grab_from_port('timestamp')
        frame_index = ts_datum.datum.get_timestamp().get_frame()

        # Debug image preparation
        disp_vis = np.clip(disp_left, 0, self.num_disp)
        disp_vis = (disp_vis / self.num_disp * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        disp_color[disp_left <= 0] = [0, 0, 0]

        rect_dets2 = []
        for j, track2 in enumerate(out_track_set2):
            frame_ids = list(track2.all_frame_ids())
            if not frame_ids:
                continue

            first_frame_id = min(frame_ids)
            first_state = track2.find_state(first_frame_id)
            if first_state is None:
                continue

            first_det = first_state.detection()

            orig_id = None
            for note in first_det.notes:
                if note.startswith("orig_id_"):
                    orig_id = int(note.split("_")[2])
                    break

            if orig_id is None:
                orig_id = track2.id
                first_det.add_note(f"orig_id_{orig_id}")

                track2.id = orig_id + self.ID_OFFSET

            if orig_id in self.locked_matches:
                track2.id = self.locked_matches[orig_id]

            state2 = track2.find_state(frame_index)
            if state2 is None: continue
            h2, t2 = self.extract_head_tail(state2.detection())
            if h2 is None or t2 is None: continue

            h2_r = self.rectify_point(h2, self.K2, self.D2, self.R2, self.P2)
            t2_r = self.rectify_point(t2, self.K2, self.D2, self.R2, self.P2)

            rect_dets2.append({
                'idx': j, 'track': track2, 'det': state2.detection(), 'orig_id': orig_id,
                'h': h2, 't': t2, 'h_r': h2_r, 't_r': t2_r
            })

        matched_j = set()

        for i, track1 in enumerate(out_track_set1):
            state1 = track1.find_state(frame_index)
            if state1 is None: continue
            det1 = state1.detection()
            h1, t1 = self.extract_head_tail(det1)
            if h1 is None or t1 is None: continue

            h1_r = self.rectify_point(h1, self.K1, self.D1, self.R1, self.P1)
            t1_r = self.rectify_point(t1, self.K1, self.D1, self.R1, self.P1)

            best_j = -1
            min_err = float('inf')
            best_det2 = None

            for det2_info in rect_dets2:
                if det2_info['idx'] in matched_j:
                    continue

                y_err_head = abs(h1_r[1] - det2_info['h_r'][1])
                y_err_tail = abs(t1_r[1] - det2_info['t_r'][1])
                total_y_err = y_err_head + y_err_tail

                disp_head = h1_r[0] - det2_info['h_r'][0]
                if total_y_err < self.matching_tolerance and disp_head > -30 and total_y_err < min_err:
                    min_err = total_y_err
                    best_j = det2_info['idx']
                    best_det2 = det2_info

            if best_j != -1:
                matched_j.add(best_j)

                det2 = best_det2['det']
                track2: Track = best_det2['track']
                orig_id = best_det2['orig_id']

                if orig_id not in self.locked_matches:
                    self.locked_matches[orig_id] = track1.id
                    track2.id = track1.id

                vector = t1_r - h1_r
                num_samples = self.samples
                samples = []

                for idx in range(num_samples):
                    f = idx / float(num_samples - 1)
                    sample_pt_2d = h1_r + f * vector
                    pt3d, disp = self.get_sample_3d(sample_pt_2d, disp_left)

                    if pt3d is not None:
                        samples.append({ 'f': f, 'pt2d': sample_pt_2d, 'pt3d': pt3d, 'disp': disp })

                if len(samples) >= 3:
                    # Theil-Sen Estimator
                    slopes = []
                    for idx1 in range(len(samples)):
                        for idx2 in range(idx1 + 1, len(samples)):
                            df = samples[idx2]['f'] - samples[idx1]['f']
                            if df > 0:
                                dz = samples[idx2]['pt3d'][2] - samples[idx1]['pt3d'][2]
                                slopes.append(dz / df)

                    robust_slope_z = np.median(slopes) if slopes else 0

                    intercepts = [s['pt3d'][2] - robust_slope_z * s['f'] for s in samples]
                    robust_intercept_z = np.median(intercepts)

                    for s in samples:
                        predicted_z = robust_slope_z * s['f'] + robust_intercept_z
                        s['error'] = abs(s['pt3d'][2] - predicted_z)

                    errors = [s['error'] for s in samples]
                    q1, q3 = np.percentile(errors, 25), np.percentile(errors, 75)
                    iqr_error = q3 - q1

                    median_z = np.median([s['pt3d'][2] for s in samples])
                    error_tolerance = max(1.2 * iqr_error, 0.05 * median_z)

                    for s in samples:
                        s['is_inlier'] = (s['error'] <= error_tolerance)

                    inliers = [s for s in samples if s['error'] <= error_tolerance]
                    outliers = [s for s in samples if s['error'] > error_tolerance]

                    if len(outliers) <= self.max_outliers:
                        F_vals = np.array([s['f'] for s in inliers])
                        X_vals = np.array([s['pt3d'][0] for s in inliers])
                        Y_vals = np.array([s['pt3d'][1] for s in inliers])
                        Z_vals = np.array([s['pt3d'][2] for s in inliers])

                        poly_x = np.polyfit(F_vals, X_vals, 1)
                        poly_y = np.polyfit(F_vals, Y_vals, 1)
                        poly_z = np.polyfit(F_vals, Z_vals, 1)

                        head_3d_robust = np.array([poly_x[1], poly_y[1], poly_z[1]])
                        tail_3d_robust = np.array([poly_x[0] + poly_x[1], poly_y[0] + poly_y[1], poly_z[0] + poly_z[1]])

                        track2.id = track1.id
                        final_length = float(np.linalg.norm(head_3d_robust - tail_3d_robust))

                        track1.set_attribute("length", final_length)
                        track2.set_attribute("length", final_length)

                        final_length_display = f"{final_length:.2f}"
                        det1.add_note(f"(atr) length {final_length_display}")
                        det2.add_note(f"(atr) length {final_length_display}")

                        # --- DEBUG DRAWING ---
                        cv2.line(disp_color, (int(h1_r[0]), int(h1_r[1])), (int(t1_r[0]), int(t1_r[1])), (255, 255, 255), 1)

                        for s in samples:
                            pt = (int(s['pt2d'][0]), int(s['pt2d'][1]))
                            color = (0, 255, 0) if s['is_inlier'] else (0, 0, 255)
                            thick = 1 if s['is_inlier'] else 2
                            cv2.circle(disp_color, pt, self.search_radius, color, thick)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        info_text = f"ID:{track1.id} Len:{final_length:.2f}m ({len(inliers)}/{num_samples} inliers)"

                        val_width_est = 35
                        spacing = 5
                        box_width = max(250, num_samples * (val_width_est + spacing)) + 10
                        box_height = 40
                        bx = int(h1_r[0]) - (box_width // 2)
                        by = int(h1_r[1]) - box_height - self.search_radius - 10
                        bx = max(5, min(bx, disp_color.shape[1] - box_width - 5))
                        by = max(5, min(by, disp_color.shape[0] - box_height - 5))
                        overlay = disp_color.copy()
                        cv2.rectangle(overlay, (bx, by), (bx + box_width, by + box_height), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, disp_color, 0.4, 0, disp_color)
                        cv2.rectangle(disp_color, (bx, by), (bx + box_width, by + box_height), (255, 255, 255), 1)
                        cv2.putText(disp_color, info_text, (bx + 5, by + 15), font, 0.45, (255, 255, 255), 1)
                        cx = bx + 5
                        cy = by + 32
                        for s in samples:
                            val_text = f"{s['disp']:.1f}"
                            color = (0, 255, 0) if s['is_inlier'] else (0, 0, 255)
                            cv2.putText(disp_color, val_text, (cx, cy), font, 0.4, color, 1)
                            cx += val_width_est + spacing

                        continue

            track1.set_attribute("length", 0)
            det1.add_note("(atr) length 0")
            if best_j != -1:
                track2.set_attribute("length", 0)
                det2.add_note("(atr) length 0")

        os.mkdir("debugDisparity")
        cv2.imwrite(f"debugDisparity/disp_frame_{frame_index:06d}.png", disp_color)

        self.push_to_port_using_trait('object_track_set_out1', ObjectTrackSet(out_track_set1))
        self.push_to_port_using_trait('object_track_set_out2', ObjectTrackSet(out_track_set2))
        self._base_step()

def __sprokit_register__():

    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python_' + __name__

    if process_factory.is_process_module_loaded(module_name):
        return

    for name, doc, cls in TMP_SPROKIT_PROCESS_REGISTRY:
        process_factory.add_process(name, doc, cls)

    process_factory.mark_process_module_as_loaded(module_name)
