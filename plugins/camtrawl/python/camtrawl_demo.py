# -*- coding: utf-8 -*-
"""
Runs camtrawl algos with no dependency on kwiver
"""
from __future__ import division, print_function
from imutils import (imscale, overlay_heatmask, putMultiLineText)
from os.path import expanduser, basename, join
import cv2
import numpy as np
import ubelt as ub

import camtrawl_algos as ctalgo


class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    def draw_detections(img, detections, masks):
        """
        Draws heatmasks showing where detector was triggered.
        Bounding boxes and contours are drawn over accepted detections.
        """
        # Upscale masks to original image size
        dsize = tuple(img.shape[0:2][::-1])
        shape = img.shape[0:2]

        # Align all masks with the image size
        masks2 = {
            k: cv2.resize(v, dsize, interpolation=cv2.INTER_NEAREST)
            for k, v in masks.items()
        }

        # Create a heatmap for detections
        draw_mask = np.zeros(shape, dtype=np.float)

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

        # Draw bounding boxes and contours
        for detection in detections:
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = np.round(detection.box_points()).astype(np.int)
            hull_points = detection.hull()
            draw_img = cv2.drawContours(
                image=draw_img, contours=[hull_points], contourIdx=-1,
                color=(255, 0, 0), thickness=2)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[box_points], contourIdx=-1,
                color=(0, 255, 0), thickness=2)
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
        import textwrap
        BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        accepted_color = BGR_RED
        rejected_color = BGR_PURPLE

        draw1 = DrawHelper.draw_detections(img1, detections1, masks1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2)
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

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_,
                                   color=rejected_color, lineType=cv2.LINE_AA,
                                   thickness=1)
                text = textwrap.dedent(
                    '''
                    error = {error:.2f}
                    '''
                ).strip().format(error=error)

                stacked = putMultiLineText(stacked, text, org=center2_,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=rejected_color,
                                           thickness=2, lineType=cv2.LINE_AA)

            # Draw accepted assignments
            for (i, j), info in zip(assignment, assign_data):
                detection1 = detections1[i]
                detection2 = detections2[j]
                center1 = np.array(detection1.oriented_bbox().center)
                center2 = np.array(detection2.oriented_bbox().center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_,
                                   color=accepted_color, lineType=cv2.LINE_AA,
                                   thickness=2)

                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}mm
                    error = {error:.2f}
                    range = {range:.2f}mm
                    '''
                ).strip().format(**info)

                stacked = putMultiLineText(stacked, text, org=center1,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=accepted_color,
                                           thickness=2, lineType=cv2.LINE_AA)

        with_orig = False
        if with_orig:
            # Put in the original images
            bottom = np.hstack([img1, img2])
            stacked = np.vstack([stacked, bottom])

        return stacked


class FrameStream(object):
    """
    Helper for iterating through a sequence of image frames.
    This will be replaced by KWIVER.
    """
    def __init__(stream, image_path_list, stride=1):
        stream.image_path_list = image_path_list
        stream.stride = stride
        stream.length = len(image_path_list)

    def __len__(stream):
        return stream.length

    def __getitem__(stream, index):
        img_fpath = stream.image_path_list[index]
        frame_id = basename(img_fpath).split('_')[0]
        img = cv2.imread(img_fpath)
        return frame_id, img

    def __iter__(stream):
        for i in range(0, len(stream), stream.stride):
            yield stream[i]


def demodata_input(dataset='test'):
    """
    Specifies the input files for testing and demos
    """
    import glob

    if dataset == 'test':
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 'haul83':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        # cal_fpath = join(data_fpath, 'code/Calib_Results_stereo_1608.mat')
        # cal_fpath = join(data_fpath, 'code/cal_201608.mat')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    else:
        assert False, 'bad dataset'

    image_path_list1 = sorted(glob.glob(join(img_path1, '*.jpg')))
    image_path_list2 = sorted(glob.glob(join(img_path2, '*.jpg')))
    assert len(image_path_list1) == len(image_path_list2)
    return image_path_list1, image_path_list2, cal_fpath


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
    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)

    detector1 = ctalgo.GMMForegroundObjectDetector()
    detector2 = ctalgo.GMMForegroundObjectDetector()
    for frame_num, (img_fpath1, img_fpath2) in enumerate(zip(image_path_list1,
                                                             image_path_list2)):

        frame_id1 = basename(img_fpath1).split('_')[0]
        frame_id2 = basename(img_fpath2).split('_')[0]
        assert frame_id1 == frame_id2
        frame_id = frame_id1
        img1 = cv2.imread(img_fpath1)
        img2 = cv2.imread(img_fpath2)

        if frame_num == target_frame_num:
            if target_step == 'detect':
                return detector1, img1

        detections1 = detector1.detect(img1)
        detections2 = detector2.detect(img2)
        masks1 = detector1._masks
        masks2 = detector2._masks

        n_detect1, n_detect2 = len(detections1), len(detections2)
        print('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2)
            dpath = ub.ensuredir('out')
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)
            break

    return detections1, detections2, cal


def demo():
    """
    Runs the algorithm end-to-end.
    """
    # dataset = 'test'
    dataset = 'haul83'

    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    dpath = ub.ensuredir('out_{}'.format(dataset))

    # ----
    # Choose parameter configurations
    # ----

    # Use GMM based model
    gmm_params = {
        # 'n_training_frames': 9999,
        # 'gmm_thresh': 30,
        # 'edge_trim': [40, 40],
        # 'n_startup_frames': 3,
        # 'factor': 1,
        # 'smooth_ksize': None,
    }
    triangulate_params = {
        # 'max_err': [6, 14],
    }
    stride = 1

    DRAWING = 0

    # ----
    # Initialize algorithms
    # ----

    detector1 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    detector2 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    triangulator = ctalgo.FishStereoMeasurments(**triangulate_params)

    import pprint
    print('dataset = {!r}'.format(dataset))
    print('Detector1 Config: ' + pprint.pformat(detector1.config, indent=4))
    print('Detector2 Config: ' + pprint.pformat(detector2.config, indent=4))
    print('Triangulate Config: ' + pprint.pformat(triangulator.config, indent=4))
    pprint.pformat(detector2.config)

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)
    stream1 = FrameStream(image_path_list1, stride=stride)
    stream2 = FrameStream(image_path_list2, stride=stride)

    # ----
    # Run the algorithm
    # ----

    n_total = 0
    all_errors = []
    all_lengths = []

    prog = ub.ProgIter(enumerate(zip(stream1, stream2)),
                       clearline=True,
                       length=len(stream1) // stride,
                       adjust=False)
    _iter = prog

    print('begin iteration')
    for frame_num, ((frame_id1, img1), (frame_id2, img2)) in _iter:
        assert frame_id1 == frame_id2
        frame_id = frame_id1

        detections1 = list(detector1.detect(img1))
        detections2 = list(detector2.detect(img2))
        masks1 = detector1._masks
        masks2 = detector2._masks

        if len(detections1) > 0 or len(detections2) > 0:
            assignment, assign_data, cand_errors = triangulator.find_matches(
                cal, detections1, detections2)
            all_errors += [d['error'] for d in assign_data]
            all_lengths += [d['fishlen'] for d in assign_data]
            n_total += len(assignment)
            # if len(assignment):
            #     prog.ensure_newline()
            #     print('n_total = {!r}'.format(n_total))
        else:
            cand_errors = None
            assignment, assign_data = None, None

        if DRAWING:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2,
                                                        assignment, assign_data,
                                                        cand_errors)
            stacked = cv2.putText(stacked,
                                  text='frame {}'.format(frame_id),
                                  org=(10, 50),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=1, color=(255, 0, 0),
                                  thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)

    all_errors = np.array(all_errors)
    all_lengths = np.array(all_lengths)
    print('n_total = {!r}'.format(n_total))
    print('ave_error = {:.2f} +- {:.2f}'.format(all_errors.mean(), all_errors.std()))
    print('ave_lengths = {:.2f} +- {:.2f} '.format(all_lengths.mean(), all_lengths.std()))


if __name__ == '__main__':
    r"""
    CommandLine:

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
        export PYTHONPATH=$(pwd):$PYTHONPATH

        python ~/code/VIAME/plugins/camtrawl/python/camtrawl_demo.py

        ffmpeg -y -f image2 -i out_haul83/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi
    """
    import camtrawl_demo
    camtrawl_demo.demo()
