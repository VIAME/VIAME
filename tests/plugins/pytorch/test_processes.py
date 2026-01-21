# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Tests for the SRNN tracker and pytorch descriptor processes.

As a minor issue, all the code to be tested here is in KWIVER
processes, which would make KWIVER's embedded pipeline functionality
the preferred way to test them.  However, said functionality isn't yet
wrapped for Python, so we're left invoking "viame" as a subprocess.

"""

import itertools
import os.path
import subprocess
import tempfile
import textwrap

from PIL import Image
import torch
import torchvision

from viame.pytorch.srnn import models


def run_pipeline_in_dir(dir_path, pipeline):
    """Run the provided pipeline file contents with the provided directory
    as current

    """
    f = tempfile.NamedTemporaryFile('w', suffix='.pipe', delete=False)
    try:
        with f:
            f.write(pipeline)
        args = ["viame", f.name]
        return subprocess.run(args, cwd=dir_path, check=True)
    finally:
        os.remove(f.name)


DESCRIPTOR_PIPELINE_TEMPLATE = textwrap.dedent("""
    config _scheduler
        type = pythread_per_process

    process images :: frame_list_input
        image_list_file = image_list.txt
        image_reader:type = ocv

    process detections :: image_object_detector
        detector:type = example_detector
        block detector:example_detector
            dx = 50
            dy = 30
        endblock

    process descriptors :: pytorch_descriptors
        gpu_list = None
        model_arch = {network}
        model_path = model.pt

    connect from images.image to detections.image
    connect from images.image to descriptors.image
    connect from detections.detected_object_set
            to descriptors.detected_object_set
""")


def build_descriptor_pipeline(network, timestamp=True):
    DPT = DESCRIPTOR_PIPELINE_TEMPLATE
    return DPT.format(network=network) + (
        "connect from images.timestamp to descriptors.timestamp\n"
        if timestamp else ''
    )


def create_test_image_list(dir_path):
    """Create image.png and image_list.txt that references it in the given
    directory

    """
    im = Image.effect_mandelbrot((800, 600), (-2.23845, -1.1538375, 0.83845, 1.1538375), 64)
    im.save(os.path.join(dir_path, 'image.png'))
    with open(os.path.join(dir_path, 'image_list.txt'), 'w') as f:
        f.writelines(itertools.repeat('image.png\n', 10))


def _test_descriptors(model_func, network, timestamp=True):
    """Run a test pipeline for the given type of feature extractor

    See the test_*_descriptors functions for how this is used.

    """
    with tempfile.TemporaryDirectory() as dir_:
        def j(*args): return os.path.join(dir_, *args)
        # Create all files required by the pipeline file
        torch.save(model_func().state_dict(), j('model.pt'))
        create_test_image_list(dir_)
        pipeline = build_descriptor_pipeline(network, timestamp=timestamp)
        run_pipeline_in_dir(dir_, pipeline)


def test_alexnet_descriptors():
    _test_descriptors(torchvision.models.alexnet, 'alexnet')


def test_resnet_descriptors():
    _test_descriptors(torchvision.models.resnet50, 'resnet')


def test_desc_augmentation():
    """Test the desc_augmentation process which uses resnet_model_path config."""
    with tempfile.TemporaryDirectory() as dir_:
        def j(*args): return os.path.join(dir_, *args)
        torch.save(torchvision.models.resnet50().state_dict(), j('model.pt'))
        create_test_image_list(dir_)
        pipeline = textwrap.dedent("""
            config _scheduler
                type = pythread_per_process

            process images :: frame_list_input
                image_list_file = image_list.txt
                image_reader:type = ocv

            process detections :: image_object_detector
                detector:type = example_detector
                block detector:example_detector
                    dx = 50
                    dy = 30
                endblock

            process descriptors :: desc_augmentation
                gpu_list = None
                resnet_model_path = model.pt

            connect from images.image to detections.image
            connect from images.image to descriptors.image
            connect from detections.detected_object_set
                    to descriptors.detected_object_set
        """)
        run_pipeline_in_dir(dir_, pipeline)


TUT_PIPELINE_TEMPLATE = textwrap.dedent("""
    config _scheduler
        type = pythread_per_process

    process images :: frame_list_input
        image_list_file = image_list.txt
        image_reader:type = ocv

    process detections :: image_object_detector
        detector:type = example_detector
        block detector:example_detector
            dx = 50
            dy = 30
        endblock

    process tracker
        :: track_objects
        :track_objects:type                       srnn

    block track_objects:srnn
        :gpu_list                                 None
        :siamese_model_path                       siamese_model.pt
        :targetRNN_AIM_model_path                 lstm_model.pt
        :targetRNN_AIM_V_model_path               lstm_model.pt
        :IOU_tracker_flag                         {iou_tracking}
    endblock

    connect from images.image to detections.image
    connect from images.image to tracker.image
    connect from detections.detected_object_set
            to tracker.detected_object_set
    connect from images.timestamp to tracker.timestamp
""")


def test_srnn_tracker():
    with tempfile.TemporaryDirectory() as dir_:
        def j(*args): return os.path.join(dir_, *args)
        # Create all files required by the pipeline file
        sm = torch.nn.DataParallel(models.Siamese())
        torch.save(dict(state_dict=sm.state_dict()), j('siamese_model.pt'))
        lm = models.TargetLSTM()
        torch.save(dict(state_dict=lm.state_dict()), j('lstm_model.pt'))
        del sm, lm
        create_test_image_list(dir_)
        for iou_tracking in [True, False]:
            pipe = TUT_PIPELINE_TEMPLATE.format(iou_tracking=iou_tracking)
            run_pipeline_in_dir(dir_, pipe)
