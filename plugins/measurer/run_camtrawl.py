# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import exists
import glob
import numpy as np
import define_pipeline
from os.path import expanduser, join, basename
import logging
import os

logging.basicConfig(level=getattr(logging, os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'INFO').upper(), logging.DEBUG))
log = logging.getLogger(__name__)
print = log.info


def make_image_input_files(data_fpath, img_path1, img_path2, start_frame=0,
                           end_frame=np.inf):
    # left_fpath = join(data_fpath, 'image_data/left')
    # right_fpath = join(data_fpath, 'image_data/right')
    cam1_image_fpaths = sorted(glob.glob(join(img_path1, '*.jpg')))
    cam2_image_fpaths = sorted(glob.glob(join(img_path2, '*.jpg')))

    def _parse_frame_id(img_fpath):
        frame_id = int(basename(img_fpath).split('_')[0])
        return frame_id

    frame_ids1 = set(map(_parse_frame_id, cam1_image_fpaths))
    frame_ids2 = set(map(_parse_frame_id, cam2_image_fpaths))
    index_lookup1 = {fid: idx for idx, fid in enumerate(frame_ids1)}
    index_lookup2 = {fid: idx for idx, fid in enumerate(frame_ids2)}

    common_frame_ids = np.array(sorted(frame_ids1.intersection(frame_ids2)))
    common_frame_ids = common_frame_ids[common_frame_ids >= start_frame]
    common_frame_ids = common_frame_ids[common_frame_ids <= end_frame]

    idxs1 = [index_lookup1[fid] for fid in common_frame_ids]
    idxs2 = [index_lookup2[fid] for fid in common_frame_ids]

    cam1_image_fpaths = [cam1_image_fpaths[idx] for idx in idxs1]
    cam2_image_fpaths = [cam2_image_fpaths[idx] for idx in idxs2]

    with open(join(data_fpath, 'cam1_images.txt'), 'w') as file:
        file.write('\n'.join(cam1_image_fpaths))

    with open(join(data_fpath, 'cam2_images.txt'), 'w') as file:
        file.write('\n'.join(cam2_image_fpaths))


def simple_pipeline():
    """
    Processing_with_species_id.m is their main file

    OpenCV2:
        cd ~/code/VIAME/plugins/camtrawl/python
        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export KWIVER_DEFAULT_LOG_LEVEL=info
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py --dataset=demo

    OpenCV3:
        cd ~/code/VIAME/plugins/camtrawl/python
        workon_py2
        source ~/code/VIAME/build-cv3-py2/install/setup_viame.sh
        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export KWIVER_DEFAULT_LOG_LEVEL=info
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py --dataset=demo

        /home/joncrall/code/VIAME/build-cv3-py2/install/bin/kwiver runner /home/joncrall/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe -S pythread_per_process
    """

    # Setup the input files
    import ubelt as ub
    dataset = ub.argval('--dataset', default='demo')

    if dataset == 'demo':
        import zipfile
        from os.path import commonprefix
        dpath = ub.ensure_app_cache_dir('camtrawl')
        try:
            demodata_zip = ub.grabdata('http://acidalia:8000/data/camtrawl_demodata.zip', dpath=dpath)
        except Exception:
            raise ValueError(
                'Demo data is currently only available on Kitware VPN')
        with zipfile.ZipFile(demodata_zip) as zfile:
            dname = commonprefix(zfile.namelist())
            data_fpath = join(dpath, dname)
            if not exists(data_fpath):
                zfile.extractall(dpath)

        print('data_fpath = {!r}'.format(data_fpath))

        cal_fpath = join(data_fpath, 'cal.npz')
        datakw = {
            'data_fpath': data_fpath,
            'img_path1': join(data_fpath, 'left'),
            'img_path2': join(data_fpath, 'right'),
        }
        print('datakw = {!r}'.format(datakw))
    elif dataset == 'test':
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        datakw = {
            'data_fpath': data_fpath,
            'img_path1': join(data_fpath, 'image_data/left'),
            'img_path2': join(data_fpath, 'image_data/right'),
        }
    elif dataset == 'haul83-small':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data_small')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        datakw = {
            'data_fpath': data_fpath,
            'img_path1': join(data_fpath, 'Haul_83/left'),
            'img_path2': join(data_fpath, 'Haul_83/right'),
        }
    elif dataset == 'haul83':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        datakw = {
            'data_fpath': data_fpath,
            'img_path1': join(data_fpath, 'Haul_83/D20160709-T021759/images/AB-800GE_00-0C-DF-06-40-BF'),  # left
            'img_path2': join(data_fpath, 'Haul_83/D20160709-T021759/images/AM-800GE_00-0C-DF-06-20-47'),  # right
            'start_frame': 2000,
            'end_frame': 5000,
        }
    else:
        import argparse
        import os
        parser = argparse.ArgumentParser(description='Camtrawl pipline demo')
        parser.add_argument('--cal', help='path to matlab or numpy stereo calibration file', default='cal.npz')
        parser.add_argument('--left', help='path to directory containing left images', default='left')
        parser.add_argument('--right', help='path to directory containing right images', default='right')
        args = parser.parse_args()
        config = args.__dict__.copy()
        img_path1, img_path2, cal_fpath = ub.take(config, [
            'left', 'right', 'cal'])
        data_fpath = os.path.dirname(img_path1)
        datakw = {
            'data_fpath': data_fpath,
            'img_path1': img_path1,
            'img_path2': img_path2,
        }
        if not exists(img_path1):
            raise IOError('left image path {!r} does not exist'.format(img_path1))
        if not exists(img_path2):
            raise IOError('right image path {!r} does not exist'.format(img_path2))
        if not exists(cal_fpath):
            raise IOError('calibration file path {!r} does not exist'.format(cal_fpath))

        # raise KeyError('Unknown dataset {}'.format(dataset))

    make_image_input_files(**datakw)

    def add_stereo_camera_branch(pipe, prefix):
        """
        Helper that defines a single branch, so it can easilly be duplicated.
        """
        image_list_file = join(data_fpath, prefix + 'images.txt')
        cam = {}

        # --- Node ---
        cam['imread'] = imread = pipe.add_process(
            name=prefix + 'imread', type='frame_list_input',
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
            'image': imread.oports['image'],
            # 'image_file_name': imread.oports['image_file_name'],
        })
        # ------------
        return cam

    pipe = define_pipeline.Pipeline()
    cam1 = add_stereo_camera_branch(pipe, 'cam1_')
    cam2 = add_stereo_camera_branch(pipe, 'cam2_')

    # stereo_cameras = pipe.add_process(
    #     name='stereo_cameras', type='stereo_calibration_camera_reader',
    #     config={
    #         # 'cal_fpath': cal_fpath,
    #     })

    # ------
    pipe.add_process(name='measure', type='camtrawl_measure', config={
        'cal_fpath': cal_fpath,
        'output_fpath': './camtrawl_out.csv',
    })
    pipe['measure'].iports.connect({
        # 'camera1': stereo_cameras.oports['camera1'],
        # 'camera2': stereo_cameras.oports['camera2'],
        'detected_object_set1': cam1['detect'].oports['detected_object_set'],
        'detected_object_set2': cam2['detect'].oports['detected_object_set'],
        'image_file_name1': cam1['imread'].oports['image_file_name'],
        'image_file_name2': cam2['imread'].oports['image_file_name'],
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
        export KWIVER_DEFAULT_LOG_LEVEL=DEBUG
        export PYTHONPATH=$HOME/code/VIAME/plugins/camtrawl:$PYTHONPATH
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

        python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py

    Ignore:
        kwiver runner ~/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe -S pythread_per_process

    Testing:
        unlink $HOME/code/VIAME/build
        ln -s $HOME/code/VIAME/build-py2.7 $HOME/code/VIAME/build

        # export SPROKIT_MODULE_PATH=$(pwd):$SPROKIT_MODULE_PATH

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export KWIVER_DEFAULT_LOG_LEVEL=info

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
