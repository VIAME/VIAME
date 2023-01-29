#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs opencv algos with no dependency on kwiver
"""
from __future__ import division, print_function, unicode_literals

from os.path import expanduser, basename, join
from os.path import exists
from functools import partial

import textwrap
import glob
import cv2
import numpy as np
import ubelt as ub
import threading
import logging

from .ocv_stereo_utils import (imscale, overlay_heatmask, putMultiLineText)
from . import ocv_stereo_algos as ctalgo
from six.moves import zip, range

logger = logging.getLogger(__name__)

try:
    import queue
except ImportError:
    # python 2 support
    import Queue as queue

class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    def draw_detections(img, detections, masks, assigned={}):
        """
        Draws heatmasks showing where detector was triggered.
        Bounding boxes and contours are drawn over accepted detections.

        Args:
            img (ndarray) :
            detections (list) :
            mask (list) :
            assigned (dict) :

        CommandLine:
            cd ~/code/VIAME/build
            source install/setup_viame.sh

            cd ~/code/VIAME/plugins/opencv/python/
            python -m xdoctest viame.processes.opencv.demo DrawHelper.draw_detections --show

        Example:
            >>> from viame.processes.opencv.algos import *
            >>> from viame.processes.opencv.demo import *
            >>> cc = np.zeros((110, 110), dtype=np.uint8)
            >>> cc[30:50, 20:70] = 1
            >>> detections = [DetectedObject.from_connected_component(cc)]
            >>> img = np.random.rand(*cc.shape)
            >>> masks = {}
            >>> assigned = {}
            >>> draw_img = DrawHelper.draw_detections(img, detections, masks, assigned)
            >>> # xdoc: +REQUIRES(--show)
            >>> fpath = ub.ensure_app_cache_dir('opencv') + '/DrawHelper.draw_detections.png'
            >>> cv2.imwrite(fpath, draw_img)
            >>> ub.startfile(fpath)

        """
        # Upscale masks to original image size
        dsize = tuple(img.shape[0:2][::-1])
        shape = img.shape[0:2]

        # Align all masks with the image size
        masks2 = {
            k: cv2.resize(v, dsize, interpolation=cv2.INTER_NEAREST)
            for k, v in masks.items()
        }

        # Create detection heatmap
        draw_mask = np.zeros(shape, dtype=float)

        if 'orig' in masks2:
            draw_mask[masks2['orig'] > 0] = .4
        if 'local' in masks2:
            draw_mask[masks2['local'] > 0] = .65
        if 'post' in masks2:
            draw_mask[masks2['post'] > 0] = .85

        for n, detection in enumerate(detections, start=1):
            bbox = detection.bbox
            dmask, factor = imscale(detection.mask.astype(np.uint8),
                                    detection.bbox_factor)
            yslice = slice(bbox.ymin, bbox.ymax + int(detection.bbox_factor))
            xslice = slice(bbox.xmin, bbox.xmax + int(detection.bbox_factor))
            draw_mask[yslice, xslice][dmask > 0] = 1.0

        draw_img = overlay_heatmask(img, draw_mask, alpha=.7)
        BGR_GREEN = (0, 255, 0)
        BGR_BLUE = (255, 0, 0)
        # BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        # Draw bounding boxes and contours
        for i, detection in enumerate(detections):
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = np.round(detection.box_points()).astype(np.int64)
            hull_points = detection.hull()
            hull_color = BGR_BLUE
            bbox_color = BGR_GREEN if i in assigned else BGR_PURPLE

            if cv2.__version__.startswith('2'):
                # 2.4 seems to operate inplace
                cv2.drawContours(
                    image=draw_img, contours=[hull_points], contourIdx=-1,
                    color=hull_color, thickness=2)

                cv2.drawContours(
                    image=draw_img, contours=[box_points], contourIdx=-1,
                    color=bbox_color, thickness=2)
            else:
                draw_img = cv2.drawContours(
                    image=draw_img, contours=[hull_points], contourIdx=-1,
                    color=hull_color, thickness=2)
                draw_img = cv2.drawContours(
                    image=draw_img, contours=[box_points], contourIdx=-1,
                    color=bbox_color, thickness=2)
        return draw_img

    @staticmethod
    def draw_stereo_detections(img1, detections1, masks1,
                               img2, detections2, masks2,
                               assignment=None, assign_data=None,
                               cand_errors=None):
        """
        Draws stereo detections side-by-side and draws lines indicating
        assignments (and near-assignments for debugging)
        """
        BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        accepted_color = BGR_RED
        rejected_color = BGR_PURPLE

        assigned1 = {} if assignment is None else [t[0] for t in assignment]
        assigned2 = {} if assignment is None else [t[1] for t in assignment]
        draw1 = DrawHelper.draw_detections(img1, detections1, masks1,
                                           assigned=assigned1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2,
                                           assigned=assigned2)
        stacked = np.hstack([draw1, draw2])

        if assignment is not None and len(cand_errors) > 0:

            # Draw candidate assignments that did not pass the thresholds
            for j in range(cand_errors.shape[1]):
                i = np.argmin(cand_errors[:, j])
                if (i, j) in assignment or (j, i) in assignment:
                    continue
                error = cand_errors[i, j]

                detection1 = detections1[i]
                detection2 = detections2[j]
                center1 = np.array(detection1.oriented_bbox().center)
                center2 = np.array(detection2.oriented_bbox().center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int64))
                center2_ = tuple(center2_.astype(np.int64))

                text = textwrap.dedent(
                    '''
                    error = {error:.2f}
                    '''
                ).strip().format(error=error)

                if cv2.__version__.startswith('2'):
                    cv2.line(stacked, center1, center2_, color=rejected_color,
                             lineType=cv2.cv.CV_AA, thickness=1)

                    stacked = putMultiLineText(stacked, text, org=center2_,
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1.5,
                                               color=rejected_color,
                                               thickness=2,
                                               lineType=cv2.cv.CV_AA)
                else:
                    stacked = cv2.line(stacked, center1, center2_,
                                       color=rejected_color, lineType=cv2.LINE_AA,
                                       thickness=1)

                    stacked = putMultiLineText(stacked, text, org=center2_,
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1.5,
                                               color=rejected_color,
                                               thickness=2,
                                               lineType=cv2.LINE_AA)

            # Draw accepted assignments
            for (i, j), info in zip(assignment, assign_data):
                detection1 = detections1[i]
                detection2 = detections2[j]
                center1 = np.array(detection1.oriented_bbox().center)
                center2 = np.array(detection2.oriented_bbox().center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int64))
                center2_ = tuple(center2_.astype(np.int64))

                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}mm
                    error = {error:.2f}
                    range = {range:.2f}mm
                    '''
                ).strip().format(**info)

                if cv2.__version__.startswith('2'):
                    cv2.line(stacked, center1, center2_, color=accepted_color,
                             lineType=cv2.cv.CV_AA, thickness=2)

                    stacked = putMultiLineText(stacked, text, org=center1,
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1.5,
                                               color=accepted_color,
                                               thickness=2,
                                               lineType=cv2.CV_AA)
                else:
                    stacked = cv2.line(stacked, center1, center2_,
                                       color=accepted_color,
                                       lineType=cv2.LINE_AA, thickness=2)

                    stacked = putMultiLineText(stacked, text, org=center1,
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1.5,
                                               color=accepted_color,
                                               thickness=2,
                                               lineType=cv2.LINE_AA)

        with_orig = False
        if with_orig:
            # Put in the original images
            bottom = np.hstack([img1, img2])
            stacked = np.vstack([stacked, bottom])

        return stacked


class FrameStream(ub.NiceRepr):
    """
    Helper for iterating through a sequence of image frames.
    This will be replaced by KWIVER.
    """
    def __init__(self, image_path_list, stride=1):
        self.image_path_list = image_path_list
        self.stride = stride
        self.length = len(image_path_list)

    def __nice__(self):
        return 'nframes={}'.format(len(self))

    def __len__(self):
        return self.length

    def _parse_frame_id(self, img_fpath):
        frame_id = basename(img_fpath).split('_')[0]
        return frame_id

    def __getitem__(self, index):
        img_fpath = self.image_path_list[index]
        frame_id = self._frame_id(img_fpath)
        logging.debug('Reading img_fpath={}'.format(img_fpath))
        img = cv2.imread(img_fpath)
        return frame_id, img

    def __iter__(self):
        for i in range(0, len(self), self.stride):
            yield self[i]


def buffered_imread(image_path_list, buffer_size=2):
    """
    Generator that runs imread in a separate thread.
    The next image is read while you are processing the current one.

    Args:
        buffer_size: the maximal number of items to pre-generate (length of the
            buffer)

    Note:
        Note: use threads instead of processes because the GIL should not
        prevent IO-bound functions like cv2.imread from running in parallel, so
        we can be reading an image while we are processing the previous one.
        (also processes have to serialize objects, which defeats the purpose of
        reading them in parallel)

    References:
        # Adapted from
        https://github.com/benanne/kaggle-ndsb/blob/11a66cdbddee16c69514b9530a727df0ac6e136f/buffering.py

        Note: a torch DataLoader might be a better option
    """
    if buffer_size < 2:
        raise ValueError('Minimal buffer size is 2')

    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
    buffer = queue.Queue(maxsize=buffer_size - 1)

    thread = threading.Thread(target=_imreader_thread,
                              args=(image_path_list, buffer))
    thread.daemon = True
    thread.start()

    # Generate the images
    data = buffer.get()
    if data is not None:
        yield data
    while data is not None:
        data = buffer.get()
        yield data


def serial_imread(image_path_list):
    for fpath in image_path_list:
        logger.debug('Reading {}'.format(fpath))
        yield cv2.imread(fpath)


def _imreader_thread(image_path_list, buffer):
    for fpath in image_path_list:
        data = cv2.imread(fpath)
        # block=True means wait until there is room on the queue
        buffer.put(data, block=True)
    buffer.put(None)  # sentinel: signal the end of the iterator


class StereoFrameStream(object):
    """
    Exmaple:
        >>> from viame.processes.opencv.demo import *
        >>> img_path1, img_path2, cal = demodata_input(dataset='haul83')
        >>> self = StereoFrameStream(img_path1, img_path2)
        >>> self.preload()
        >>> # -------
        >>> prog_thread = ub.ProgIter(self._stream(buffer_size=2), label='thread', length=len(self), adjust=False, freq=1)
        >>> for _ in prog_thread:
        >>>     pass
        >>> print('prog_thread._total_seconds = {!r}'.format(prog_thread._total_seconds))
        >>> # -------
        >>> prog_serial = ub.ProgIter(self._stream(buffer_size=None), label='serial', length=len(self), adjust=False, freq=1)
        >>> for _ in prog_serial:
        >>>     pass
        >>> print('prog_serial._total_seconds = {!r}'.format(prog_serial._total_seconds))
    """
    def __init__(self, img_path1, img_path2, buffer_size=1):
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.image_path_list1 = None
        self.image_path_list2 = None
        self.aligned_frameids = []
        self.aligned_idx1 = None
        self.aligned_idx2 = None
        self.index = 0
        self.buffer_size = buffer_size

    def seek(self, index):
        logging.debug('seek {}'.format(index))
        if index > len(self.aligned_frameids):
            raise IndexError('Out of bounds')
        self.index = index

    def __len__(self):
        return len(self.aligned_frameids) - self.index

    def _stream(self, buffer_size=None):
        if buffer_size is None:
            buffer_size = self.buffer_size
        if buffer_size and buffer_size > 1:
            logging.debug('Reading image stream using thread buffer')
            buffer_func = partial(buffered_imread, buffer_size=buffer_size)
        else:
            logging.debug('Reading image stream using serial buffer')
            buffer_func = serial_imread

        idx1_slice = self.aligned_idx1[self.index:]
        idx2_slice = self.aligned_idx2[self.index:]
        frame_ids = self.aligned_frameids[self.index:]

        fpath_gen1 = ub.take(self.image_path_list1, idx1_slice)
        fpath_gen2 = ub.take(self.image_path_list2, idx2_slice)

        img_gen1 = buffer_func(fpath_gen1)
        img_gen2 = buffer_func(fpath_gen2)

        logging.debug('Begin reading buffers')
        for frame_id, img1, img2 in zip(frame_ids, img_gen1, img_gen2):
            yield frame_id, img1, img2
            self.index += 1
        logging.debug('Stream is finished')

    def __iter__(self):
        for data in self._stream():
            yield data

    def _parse_frame_id(self, img_fpath):
        frame_id = basename(img_fpath).split('_')[0]
        return frame_id

    def preload(self):
        self.image_path_list1 = sorted(glob.glob(join(self.img_path1, '*.jpg')))
        self.image_path_list2 = sorted(glob.glob(join(self.img_path2, '*.jpg')))
        self.align_frames()

    def align_frames(self, verbose=1):
        frame_ids1 = list(map(self._parse_frame_id, self.image_path_list1))
        frame_ids2 = list(map(self._parse_frame_id, self.image_path_list2))

        n_imgs1 = len(self.image_path_list1)
        n_imgs2 = len(self.image_path_list2)

        aligned_idx1 = []
        aligned_idx2 = []
        aligned_frameids = []

        idx1, idx2 = 0, 0
        while idx1 < n_imgs1 and idx2 < n_imgs2:
            frame_id1 = frame_ids1[idx1]
            frame_id2 = frame_ids2[idx2]
            if frame_id1 == frame_id2:
                aligned_idx1.append(idx1)
                aligned_idx2.append(idx2)
                aligned_frameids.append(frame_id1)
                idx1 += 1
                idx2 += 1
            elif frame_id1 < frame_id2:
                idx1 += 1
            elif frame_id1 > frame_id2:
                idx2 += 1

        if verbose:
            # missing_from1 = set(frame_ids2) - set(frame_ids1)
            # missing_from2 = set(frame_ids1) - set(frame_ids2)
            # print('missing_from1 = {!r}'.format(sorted(missing_from1)))
            # print('missing_from2 = {!r}'.format(sorted(missing_from2)))
            n_aligned = len(aligned_idx1)
            logging.info('Camera1 aligned {}/{} frames'.format(n_aligned, n_imgs1))
            logging.info('Camera2 aligned {}/{} frames'.format(n_aligned, n_imgs2))

        self.aligned_frameids = aligned_frameids
        self.aligned_idx1 = aligned_idx1
        self.aligned_idx2 = aligned_idx2


def demodata_input(dataset='demo'):
    """
    Specifies the input files for testing and demos
    """
    if dataset == 'demo':
        import zipfile
        from os.path import commonprefix
        dpath = ub.ensure_app_cache_dir('opencv')
        try:
            demodata_zip = ub.grabdata('http://acidalia:8000/data/opencv_demodata.zip', dpath=dpath)
        except Exception:
            raise ValueError(
                'Demo data is currently only available on Kitware VPN')
        with zipfile.ZipFile(demodata_zip) as zfile:
            dname = commonprefix(zfile.namelist())
            data_fpath = join(dpath, dname)
            if not exists(data_fpath):
                zfile.extractall(dpath)

        cal_fpath = join(data_fpath, 'cal.npz')
        img_path1 = join(data_fpath, 'left')
        img_path2 = join(data_fpath, 'right')
    elif dataset == 'test':
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 'haul83-small':
        data_fpath = expanduser('~/data/opencv_stereo_sample_data_small')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    elif dataset == 'haul83':
        data_fpath = expanduser('~/data/opencv_stereo_sample_data/')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/D20160709-T021759/images/AB-800GE_00-0C-DF-06-40-BF')  # left
        img_path2 = join(data_fpath, 'Haul_83/D20160709-T021759/images/AM-800GE_00-0C-DF-06-20-47')  # right
    else:
        raise ValueError('unknown dataset={!r}'.format(dataset))
    return img_path1, img_path2, cal_fpath


def demodata_calibration():
    cal = ctalgo.StereoCalibration._from_flat_dict({
        # Left Intrinsics
        'fc_left': np.array([1, 1]),  # focal point
        'cc_left': np.array([0, 0]),  # principle point
        'alpha_c_left': [0],  # skew
        'kc_left': np.array([0, 0, 0, 0, 0]),  # distortion
        # Right Intrinsics
        'fc_right': np.array([1, 1]),  # focal point
        'cc_right': np.array([0, 0]),  # principle point
        'alpha_c_right': [0],  # skew
        'kc_right': np.array([0, 0, 0, 0, 0]),  # distortion
        # Right Extrinsics (wrt left as the origin)
        'om': np.array([0, 0, 0]),  # rotation vector
        'T': np.array([0, 1, 0]),  # translation vector
    })
    return cal


def demodata_detections2():
    """ Dummy data for tests """
    cal = demodata_calibration()

    detections1 = []
    cc_mask = np.zeros((11, 11), dtype=np.uint8)
    cc_mask[3:5, 2:7] = 1
    det1 = ctalgo.DetectedObject.from_connected_component(cc_mask)
    detections1.append(det1)

    detections2 = []
    cc_mask = np.zeros((11, 11), dtype=np.uint8)
    cc_mask[4:6, 3:8] = 1
    det2 = ctalgo.DetectedObject.from_connected_component(cc_mask)
    detections2.append(det2)

    return cal, detections1, detections2


def demodata_detections(dataset='haul83', target_step='detect', target_frame_num=7):
    """
    Helper for doctests. Gets test data at different points in the pipeline.
    """
    # <ipython hacks>
    if 'target_step' not in vars():
        target_step = 'detect'
    if 'target_frame_num' not in vars():
        target_frame_num = 7
    # </ipython hacks>
    img_path1, img_path2, cal_fpath = demodata_input(dataset=dataset)

    stream = StereoFrameStream(img_path1, img_path2)
    stream.preload()

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)

    detector1 = ctalgo.GMMForegroundObjectDetector()
    detector2 = ctalgo.GMMForegroundObjectDetector()

    for frame_num, (frame_id, img1, img2) in enumerate(stream):
        if frame_num == target_frame_num:
            if target_step == 'detect':
                return detector1, img1

        detections1 = detector1.detect(img1)
        detections2 = detector2.detect(img2)
        masks1 = detector1._masks
        masks2 = detector2._masks

        n_detect1, n_detect2 = len(detections1), len(detections2)
        logging.info('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2)
            drawing_dpath = ub.ensuredir('out')
            cv2.imwrite(drawing_dpath + '/mask{}_draw.png'.format(frame_id), stacked)
            break

    return detections1, detections2, cal


class FrozenKeyDict(dict):
    """
    Example:
        >>> self = FrozenKeyDict({1: 2, 3: 4})
    """
    def __init__(self, *args, **kwargs):
        super(FrozenKeyDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            raise ValueError('FrozenKeyDict keys cannot be modified')
        super(FrozenKeyDict, self).__setitem__(key, value)

    def clear(self, *args, **kw):
        raise ValueError('FrozenKeyDict keys cannot be modified')

    def pop(self, *args, **kw):
        raise ValueError('FrozenKeyDict keys cannot be modified')

    def __delitem__(self, *args, **kw):
        raise ValueError('FrozenKeyDict keys cannot be modified')


def demo(config=None):
    """
    Runs the algorithm end-to-end.
    """
    # dataset = 'test'
    # dataset = 'haul83'

    if config is None:
        import argparse
        parser = argparse.ArgumentParser(description='Standalone opencv demo')

        parser.add_argument('--cal', help='path to matlab or numpy stereo calibration file', default='cal.npz')
        parser.add_argument('--left', help='path to directory containing left images', default='left')
        parser.add_argument('--right', help='path to directory containing right images', default='right')
        parser.add_argument('--out', help='output directory', default='./out')
        parser.add_argument('-f', '--overwrite', action='store_true', help='will delete any existing output')
        parser.add_argument('--draw', action='store_true', help='draw visualization of algorithm steps')

        parser.add_argument('--dataset', default=None,
                            help='Developer convenience assumes you have demo '
                                 ' data downloaded and available. If you dont '
                                 ' specify the other args.')

        args = parser.parse_args()
        config = args.__dict__.copy()
        config = FrozenKeyDict(config)

    if config['dataset'] is not None:
        img_path1, img_path2, cal_fpath = demodata_input(dataset=config['dataset'])
        config['left'] = img_path1
        config['right'] = img_path2
        config['cal'] = cal_fpath

    img_path1, img_path2, cal_fpath = ub.take(config, [
        'left', 'right', 'cal'])
    out_dpath = config['out']
    logging.info('Demo Config = {!r}'.format(config))

    ub.ensuredir(out_dpath)

    # ----
    # Choose parameter configurations
    # ----

    # Use GMM based model
    gmm_params = {
    }
    triangulate_params = {
    }

    DRAWING = config['draw']

    # ----
    # Initialize algorithms
    # ----

    detector1 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    detector2 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    triangulator = ctalgo.StereoLengthMeasurments(**triangulate_params)

    try:
        import pyfiglet
        print(pyfiglet.figlet_format('CAMTRAWL', font='cybermedium'))
    except ImportError:
        logging.debug('pyfiglet is not installed')
        print('========')
        print('CAMTRAWL')
        print('========')
    logging.info('Detector1 Config: ' + ub.repr2(detector1.config, nl=1))
    logging.info('Detector2 Config: ' + ub.repr2(detector2.config, nl=1))
    logging.info('Triangulate Config: ' + ub.repr2(triangulator.config, nl=1))
    logging.info('DRAWING = {!r}'.format(DRAWING))

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)

    stream = StereoFrameStream(img_path1, img_path2)
    stream.preload()

    # HACK IN A BEGIN FRAME
    if len(stream) > 2200:
        stream.seek(2200)

    # ----
    # Run the algorithm
    # ----

    # n_frames = 2000
    # stream.aligned_frameids = stream.aligned_frameids[:stream.index]

    measure_fpath = join(out_dpath, 'measurements.csv')
    if exists(measure_fpath):
        if config['overwrite']:
            ub.delete(measure_fpath)
        else:
            raise IOError('Measurement path already exists')
    output_file = open(measure_fpath, 'a')

    if DRAWING:
        drawing_dpath = join(out_dpath, 'visual')
        if exists(drawing_dpath):
            if config['overwrite']:
                ub.delete(drawing_dpath)
            else:
                raise IOError('Output path already exists')
        ub.ensuredir(drawing_dpath)

    headers = ['current_frame', 'fishlen', 'range', 'error', 'dz', 'box_pts1',
               'box_pts2']
    output_file.write(','.join(headers) + '\n')
    output_file.flush()

    measurements = []

    logger.info('begin opencv iteration')

    import tqdm
    # prog = ub.ProgIter(iter(stream), total=len(stream), desc='opencv demo',
    #                    clearline=False, freq=1, adjust=False)
    prog = tqdm.tqdm(iter(stream), total=len(stream), desc='opencv demo',
                     leave=True)

    def csv_repr(d):
        if isinstance(d, np.ndarray):
            d = d.tolist()
        s = repr(d)
        return s.replace('\n', '').replace(',', ';').replace(' ', '')

    for frame_num, (frame_id, img1, img2) in enumerate(prog):
        logger.debug('frame_num = {!r}'.format(frame_num))

        detections1 = list(detector1.detect(img1))
        detections2 = list(detector2.detect(img2))
        masks1 = detector1._masks
        masks2 = detector2._masks

        any_detected = len(detections1) > 0 or len(detections2) > 0

        if any_detected:
            assignment, assign_data, cand_errors = triangulator.find_matches(
                cal, detections1, detections2)
            # Append assignments to the measurements
            for data in assign_data:
                data['current_frame'] = int(frame_id)
                measurements.append(data)
                line = ','.join([csv_repr(d) for d in ub.take(data, headers)])
                output_file.write(line + '\n')
                output_file.flush()
        else:
            cand_errors = None
            assignment, assign_data = None, None

        if DRAWING >= 2 or (DRAWING and any_detected):
            DRAWING = 3
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2,
                                                        assignment, assign_data,
                                                        cand_errors)
            if cv2.__version__.startswith('2'):
                cv2.putText(stacked,
                            text='frame #{}, id={}'.format(frame_num,
                                                           frame_id),
                            org=(10, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0),
                            thickness=2, lineType=cv2.cv.CV_AA)
            else:
                stacked = cv2.putText(stacked,
                                      text='frame #{}, id={}'.format(frame_num,
                                                                     frame_id),
                                      org=(10, 50),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=1, color=(255, 0, 0),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(drawing_dpath + '/mask{}_draw.png'.format(frame_id), stacked)
    output_file.close()

    n_total = len(measurements)
    logger.info('n_total = {!r}'.format(n_total))
    if n_total:
        all_errors = np.array([d['error'] for d in measurements])
        all_lengths = np.array([d['fishlen'] for d in measurements])
        logger.info('ave_error = {:.2f} +- {:.2f}'.format(all_errors.mean(), all_errors.std()))
        logger.info('ave_lengths = {:.2f} +- {:.2f} '.format(all_lengths.mean(), all_lengths.std()))
    return measurements


def setup_demo_logger():
    import logging
    import logging.config
    from os.path import exists
    import ubelt as ub

    logconf_fpath = 'logging.conf'
    if exists(logconf_fpath):
        logging.config.fileConfig(logconf_fpath)
    else:
        level = getattr(logging, ub.argval('--level', default='INFO').upper())
        logfmt = '%(levelname)s %(name)s(%(lineno)d): %(message)s'
        logging.basicConfig(format=logfmt, level=level)
    logger.debug('Setup logging in demo script')


if __name__ == '__main__':
    r"""
    Developer:
        workon_py2
        source ~/code/VIAME/build-py2.7/install/setup_viame.sh

        cd ~/code/VIAME/plugins/opencv/python/opencv

        python -m viame.processes.opencv.demo --dataset=haul83-small
        python -m viame.processes.opencv.demo --level=DEBUG

        # Create a movie rom the frames
        ffmpeg -y -f image2 -i out/visual/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi

    Python2:
        # Runs in about 2-3 it/s
        workon_py2
        source ~/code/VIAME/build-py2.7/install/setup_viame.sh
        python -m viame.processes.opencv.demo --dataset=demo --out=out-py2 -f
        python -m viame.processes.opencv.demo --dataset=demo --draw --out=out-py2

    Python3:
        # Runs in about 3-4 it/s
        workon_py3
        cd ~/code/VIAME/plugins/opencv/python
        python -m viame.processes.opencv.demo --dataset=demo --out=out-py3 -f
        python -m viame.processes.opencv.demo --dataset=demo --draw --out=out-py3


    CommandLine:
        python -m viame.processes.opencv.demo --help

        # Move to the data directory
        cd ~/data/opencv_stereo_sample_data_small

        # Run with args specific to that data directory
        python -m viame.processes.opencv.demo \
            --left=Haul_83/left --right=Haul_83/right \
            --cal=201608_calibration_data/selected/Camtrawl_2016.npz \
            --out=out_haul83 --draw
    """
    setup_demo_logger()
    measurements = demo()
