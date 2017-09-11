# -*- coding: utf-8 -*-
from __future__ import print_function
import glob
import sprokit_pipeline
from os.path import expanduser, join
import logging
import os

logging.basicConfig(level=getattr(logging, os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'INFO').upper(), logging.DEBUG))
log = logging.getLogger(__name__)
print = log.info


def make_image_input_files(data_fpath):
    left_fpath = join(data_fpath, 'image_data/left')
    right_fpath = join(data_fpath, 'image_data/right')
    cam1_image_fpaths = sorted(glob.glob(join(left_fpath, '*.jpg')))
    cam2_image_fpaths = sorted(glob.glob(join(right_fpath, '*.jpg')))

    # Just use the first n for testing
    # n = len(cam1_image_fpaths)
    n = 1
    cam1_image_fpaths = cam1_image_fpaths[0:n]
    cam2_image_fpaths = cam2_image_fpaths[0:n]

    with open(join(data_fpath, 'cam1_images.txt'), 'w') as file:
        file.write('\n'.join(cam1_image_fpaths))

    with open(join(data_fpath, 'cam2_images.txt'), 'w') as file:
        file.write('\n'.join(cam2_image_fpaths))


def simple_pipeline():
    """
    Processing_with_species_id.m is their main file

    CommandLine:
        cd ~/code/VIAME/plugins/camtrawl/python
        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py
    """

    # Setup the input files

    data_fpath = expanduser('~/data/autoprocess_test_set')
    # cal_fpath = join(data_fpath, 'cal_201608.mat')

    make_image_input_files(data_fpath)

    def add_stereo_camera_branch(pipe, prefix):
        """
        Helper that defines a single branch, so it can easilly be duplicated.
        """
        image_list_file = join(data_fpath, prefix + 'images.txt')
        cam = {}

        # --- Node ---
        input_image = pipe.add_process(
            name=prefix + 'input_image', type='frame_list_input',
            config={
                'image_list_file': image_list_file,
                'frame_time': 0.03333333,
                'image_reader:type': 'ocv',
            })
        # ------------

        # --- Node ---
        cam['detect'] = detect = pipe.add_process(
            name=prefix + 'detect', type='camtrawl_detect_fish', config={ })
        detect.iports.connect({
            'image': input_image.oports['image']
        })
        # ------------
        return cam

    pipe = sprokit_pipeline.Pipeline()
    cam1 = add_stereo_camera_branch(pipe, 'cam1_')
    cam2 = add_stereo_camera_branch(pipe, 'cam2_')

    # stereo_cameras = pipe.add_process(
    #     name='stereo_cameras', type='stereo_calibration_camera_reader',
    #     config={
    #         # 'cal_fpath': cal_fpath,
    #     })

    # ------
    pipe.add_process(name='measure', type='camtrawl_measure', config={})
    pipe['measure'].iports.connect({
        # 'camera1': stereo_cameras.oports['camera1'],
        # 'camera2': stereo_cameras.oports['camera2'],
        'detected_object_set1': cam1['detect'].oports['detected_object_set'],
        'detected_object_set2': cam2['detect'].oports['detected_object_set'],
    })
    # ------

    pipe.config['_pipeline:_edge']['capacity'] = 1
    pipe.config['_scheduler']['type'] = 'pythread_per_process'

    # pipe.draw_graph('pipeline.png')
    # import ubelt as ub
    # ub.startfile('pipeline.png')

    print('  --- RUN PIPELINE ---')
    import ubelt as ub
    with ub.Timer('Running Pipeline'):
        pipe.run()

    return pipe


if __name__ == '__main__':
    """
    CommandLine:
        # RUNNING
        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export KWIVER_DEFAULT_LOG_LEVEL=info
        export PYTHONPATH=$HOME/code/VIAME/plugins/camtrawl:$PYTHONPATH
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py

    Testing:
        # export SPROKIT_MODULE_PATH=$(pwd):$SPROKIT_MODULE_PATH

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export KWIVER_DEFAULT_LOG_LEVEL=debug
        # export KWIVER_DEFAULT_LOG_LEVEL=info

        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$HOME/code/VIAME/plugins/camtrawl:$PYTHONPATH
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes
        cd ~/code/VIAME/plugins/camtrawl/

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py

        cd ~/code/VIAME/plugins/camtrawl/
        cls && python -c "from sprokit import pipeline; print(pipeline.process_factory.types())"

        # Issue with sprokit python
        ```python
        from sprokit import pipeline

        type_list = pipeline.process_factory.types()

        # Should be vector<string> (or list of strings) but instead its ports?
        print('type_list = {!r}'.format(type_list))

        # calling help results in a SEGFAULT
        help(ports)
        ```
    """
    simple_pipeline()
