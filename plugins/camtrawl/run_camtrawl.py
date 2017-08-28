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


def read_matlab_stereo_camera(cal_fpath):
        import itertools as it
        import itertools as it
    """
    References:
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html

    Ignore:
        from os.path import expanduser
        cal_fpath = expanduser('~/data/autoprocess_test_set/cal_201608.mat')

        import sys
        sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl')
        from camtrawl_pipeline_def import *

        stereo_camera = read_matlab_stereo_camera(cal_fpath)

        intrin = stereo_camera['intrinsic_left']
        fc = intrin['fc']
        cc = intrin['cc']
        alpha_c = intrin['alpha_c']
        kc = intrin['kc']
        KK = np.array([
            [fc[0], alpha_c * fc[0], cc[0]],
            [    0,           fc[1], cc[1]],
            [    0,               0,     1],
        ])
    """
    import scipy.io
    cal_data = scipy.io.loadmat(cal_fpath)
    cal = cal_data['Cal']

    (om, T, fc_left, fc_right, cc_left, cc_right, kc_left, kc_right,
     alpha_c_left, alpha_c_right) = cal[0][0]
    stereo_camera = {
        'extrinsic': {
            'om': om.ravel(),  # rotation vector
            'T': T.ravel(),  # translation vector
        },
        'intrinsic_left': {
            'fc': fc_left.ravel(),  # focal point
            'cc': cc_left.ravel(),  # principle point
            'alpha_c': alpha_c_left.ravel()[0],  # skew
            'kc': kc_left.ravel(),  # distortion
        },
        'intrinsic_right': {
            'fc': fc_right.ravel(),
            'cc': cc_right.ravel(),
            'alpha_c': alpha_c_right.ravel()[0],
            'kc': kc_right.ravel(),
        },
    }
    return stereo_camera


def make_image_input_files(data_fpath):
    left_fpath = join(data_fpath, 'image_data/left')
    right_fpath = join(data_fpath, 'image_data/right')
    cam1_image_fpaths = sorted(glob.glob(join(left_fpath, '*.jpg')))
    cam2_image_fpaths = sorted(glob.glob(join(right_fpath, '*.jpg')))

    # Just use the first n for testing
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
        cd ~/code/VIAME/plugins/camtrawl
        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/camtrawl_pipeline_def.py
    """

    # Setup the input files

    data_fpath = expanduser('~/data/autoprocess_test_set')
    cal_fpath = join(data_fpath, 'cal_201608.mat')

    stereo_camera = read_matlab_stereo_camera(cal_fpath)  # NOQA
    make_image_input_files(data_fpath)

    def add_stereo_camera_branch(pipe, prefix):
        """
        Helper that defines a single branch, so it can easilly be duplicated.
        """
        image_list_file = join(data_fpath, prefix + 'images.txt')

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
        detect = pipe.add_process(
            name=prefix + 'detect', type='camtrawl_detect_fish', config={ })
        detect.iports.connect({
            'image': input_image.oports['image']
        })
        # ------------

        # --- Node ---
        biomarker = pipe.add_process(
            name=prefix + 'biomarker', type='camtrawl_detect_biomarker', config={ })
        biomarker.iports.connect({
            'image': input_image.oports['image'],
            'detected_object_set': detect.oports['detected_object_set'],
        })
        # ------------

    pipe = sprokit_pipeline.Pipeline()
    add_stereo_camera_branch(pipe, 'cam1_')
    add_stereo_camera_branch(pipe, 'cam2_')

    # ------
    pipe.add_process(name='measure', type='camtrawl_measure', config={})
    pipe['measure'].iports.connect({
        # 'camera1': pipe['cam1_input_camera'].oports['camera'],
        # 'camera2': pipe['cam2_input_camera'].oports['camera'],
        'feature_set1': pipe['cam1_biomarker'].oports['feature_set'],
        'feature_set2': pipe['cam2_biomarker'].oports['feature_set'],
    })
    # ------

    pipe.config['_pipeline:_edge']['capacity'] = 1
    pipe.config['_scheduler']['type'] = 'pythread_per_process'

    # pipe.draw_graph('pipeline.png')
    # import ubelt as ub
    # ub.startfile('pipeline.png')

    print('  --- RUN PIPELINE ---')
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

        python ~/code/VIAME/plugins/camtrawl/camtrawl_pipeline_def.py

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

        python ~/code/VIAME/plugins/camtrawl/camtrawl_pipeline_def.py

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
