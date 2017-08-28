# -*- coding: utf-8 -*-
"""
Prototype for a python based pipeline definition

Python Stuff lives here:
    ~/code/VIAME/packages/kwiver/sprokit/src/bindings/python
    ~/code/VIAME/packages/kwiver/vital/bindings/python/vital/types

NOTES:
    See set_input_port_frequency

Useful Places:
    ~/code/VIAME/packages/kwiver/vital/types
    ~/code/VIAME/packages/kwiver/vital/algo

    ~/code/VIAME/packages/kwiver/arrows/ocv
    ~/code/VIAME/packages/kwiver/arrows/core

    ~/code/VIAME/packages/kwiver/sprokit/processes/core
    ~/code/VIAME/packages/kwiver/sprokit/processes/kwiver_type_traits.h

Useful Types:

    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/refine_detections_process.h

    detected_object():
        ~/code/VIAME/packages/kwiver/vital/types/detected_object.h
        bounding_box_d bounding_box
        index
        confidence
        detector_name
      image_container_sptr mask();

    feature():
        # An 2D feature point in an image (with optional scale/angle/etc..)
        ~/code/VIAME/packages/kwiver/vital/types/feature.h
        typedef std::shared_ptr< feature > feature_sptr;
        vector_2d loc

    landmark():
        # An abstract representation of a 3D world point
        ~/code/VIAME/packages/kwiver/vital/types/landmark.h


Useful Type Collections:

    detected_object_set():
        ~/code/VIAME/packages/kwiver/vital/types/detected_object_set.h

    feature_set():
        ~/code/VIAME/packages/kwiver/vital/types/feature_set.h
        std::vector< feature_sptr > features

    landmark_map():
        ~/code/VIAME/packages/kwiver/vital/types/landmark_map.h
          typedef std::map< landmark_id_t, landmark_sptr > map_landmark_t;
          typedef int64_t landmark_id_t;
"""
# import numpy as np
import sprokit_pipeline


def notes_about_nodes_that_turned_out_to_be_less_useful_than_i_thought(pipe):
    """
    Ignore. These seem to not be relevant. Keeping my notes for reference.
    """
    # -----
    # DEFINATELY DONT WANT A MATCHER PROCESS.
    # Should it be renamed to sequential_matcher?
    # Is there a pairwise (one-vs-one) matcher or a (one-vs-many) matcher based
    # on an inverted index (and possibly a visual vocab / NetVLAD?)
    # arrows/vxl/match_features_constrained.h has pariwise
    pipe.add_process(name='fish_point_matcher', type='feature_matcher', config={})
    # see ~/code/VIAME/packages/kwiver/vital/algo/match_features.h
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/matcher_process.h
    # see ~/code/VIAME/packages/kwiver/arrows/ocv/match_features.h
    # iports = {timestamp, image, feature_set, descriptor_set}  # all required
    # oports = {feature_track_set}
    pass


def add_stereo_camera_branch(pipe, prefix):
    """
    Helper that defines a single branch, so it can easilly be duplicated.

    Notes:
        ~/code/VIAME/packages/kwiver/sprokit/processes/examples/process_template/template_process.cxx
    """
    # ============
    # Branch Nodes
    # ============

    # -----
    # Does this exist?
    input_camera = pipe.add_process(name=prefix + 'input_camera', type='camera_input_list', config={
        'image_list_file': 'images.txt',
        'frame_time': 0.03333333,
    })
    # read in a KRTD
    # ~/code/VIAME/packages/kwiver/vital/io/camera_io.h
    # camera_input_list.oports = {camera}

    # -----
    input_image = pipe.add_process(name=prefix + 'input_image', type='frame_input_list', config={})
    # see ~/code/VIAME/packages/kwiver/vital/algo/image_io.h
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/frame_list_process.h
    # see ~/code/VIAME/packages/kwiver/arrows/ocv/image_io.cxx
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/image_file_reader_process.h
    # NOTE: image_file_reader_process obsoletes frame_list_process?
    # frame_input_list.oports = {image, frame, time}

    # -----
    # TODO: Needs specific implementation
    fish_detector = pipe.add_process(name=prefix + 'fish_detector', type='image_object_detector', config={})
    # see ~/code/VIAME/packages/kwiver/vital/algo/image_object_detector.h
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/image_object_detector_process.h
    # see ~/code/VIAME/packages/kwiver/arrows/ocv/match_features_bruteforce.cxx
    # image_object_detector.iports = {image}
    # image_object_detector.oports = {detected_object_set}

    # -----
    # TODO: does this exist? Does this need implementation?
    chip_extractor = pipe.add_process(name=prefix + 'chip_extractor', type='chip_extractor', config={})
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/refine_detections_process.h
    # see ~/code/VIAME/packages/kwiver/vital/algo/refine_detections.h

    # chip_extractor.iports = {image, detected_object_set}
    # chip_extractor.oports = {chip}

    # -----
    # TODO: Needs specific implementation / maybe a reimplementation
    # I think it should be possible to subclass detect_features
    fish_points = pipe.add_process(name=prefix + 'fish_points', type='detect_features', config={})
    # see ~/code/VIAME/packages/kwiver/vital/algo/detect_features.h
    # see ~/code/VIAME/packages/kwiver/sprokit/processes/core/detect_features_process.h
    # detect_features.iports = {image, timestamp}
    # detect_features.oports = {feature_set}

    # ==================
    # Branch Connections
    # ==================

    fish_detector.iports.connect({
        'image': input_image.oports['image'],
    })

    chip_extractor.iports.connect({
        'image': input_image.oports['image'],
        # Does chip extractor take multiple bboxes or just one?
        # See set_input_port_frequency
        'detected_object_set': fish_detector.oports['detected_object_set'],
    })
    # fish_detector.oports['detected_object_set'].connect(
    #     chip_extractor.iports['detected_object_set'])

    fish_points.iports.connect({
        'image': chip_extractor.oports['chip']
    })

    return fish_points, input_camera


def define_fishlen_pipeline():
    """
    Defines a pipeline to run the algorithms for determening fish length using
    stereo cameras and specialized fish-point detection algorithms.
    """

    # Initialize a new pipeline.
    # The main idea is that pipeline will register and store all information,
    # and (for now) sipmly write a .pipe file that can be fed to sprokit.
    pipe = sprokit_pipeline.Pipeline()

    # =====
    # Nodes
    # =====

    # Define a branch for each camera and return references to
    # the camera and fish point nodes.

    fish_points1, input_camera1 = add_stereo_camera_branch(pipe, 'cam1_')
    fish_points2, input_camera2 = add_stereo_camera_branch(pipe, 'cam2_')

    # ----------------
    # Probably needs to be adapted
    pipe.add_process(name='triangulate_fish_points', type='triangulate_stereo_points', config={})
    # see triangulate_landmarks
    # see ~/code/VIAME/packages/kwiver/vital/algo/triangulate_landmarks.h
    # see ~/code/VIAME/packages/kwiver/arrows/core/triangulate.h
    # Can probably use
    # triangulate_inhomog(const std::vector<vital::simple_camera >& cameras,
    #                     const std::vector<Eigen::Matrix<T,2,1> >& points);
    # node.iports = {camera1, camera2, feature_set1, feature_set2, correspondence=None}
    # node.oports = {landmark}

    # ----------------
    # Does not exist yet
    pipe.add_process(name='measure_length', type='distance',
                     config={'distance_type': 'L2'})
    # How do we get
    # node.iports = {points1, points2, correspondence=None}
    # node.oports = {distance}

    # ----------------
    # Does something exist to dump the data somewhere?

    pipe.add_process(name='fishlen_output_node', type='print_number',
                     config={'output': 'fishlens.txt'})
    # ~/code/VIAME/packages/kwiver/sprokit/src/processes/examples/print_number_process.h
    # print_number.iports = {number}
    # print_number.config = {output}

    # ===========
    # Connections
    # ===========

    pipe['triangulate_fish_points'].iports.connect({
        'camera1': input_camera1.oports['camera'],
        'camera2': input_camera2.oports['camera'],
        'feature_set1': fish_points1.oports['feature_set'],
        'feature_set2': fish_points2.oports['feature_set'],
        # Optional corresponding indices between feature sets.
        # If not specified they will be assumed to be equal length and aligned.
        'correspondence': None,
    })

    pipe['measure_length'].iports.connect({
        # FIXME: find a nice way to specify a correspondence
        'points1': pipe['triangulate_fish_points'].oports['landmarks'],
        'points2': pipe['triangulate_fish_points'].oports['landmarks'],
        # 'correspondence': np.array([[0, 1]]),
    })

    pipe['fishlen_output_node'].iports.connect({
        'number': pipe['measure_length'].oports['distance']
    })

    # global pipeline config
    # pipe.config['_edge']['capacity'] = 10

    return pipe


def dummy_pipeline():

    """
    Processing_with_species_id.m is their main file


    from os.path import expanduser
    import scipy.io
    cal_fpath = expanduser('~/data/autoprocess_test_set/cal_201608.mat')
    cal_data = scipy.io.loadmat(cal_fpath)
    cal = cal_data['Cal']

    # References:
    #    http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
    #    http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
    (om, T, fc_left, fc_right, cc_left, cc_right, kc_left, kc_right,
    alpha_c_left, alpha_c_right) = cal[0][0]

    extrinsic_stereo_params = {
        'om': om,  # rotation vector
        'T': T,  # translation vector
    }

    intrinsic_left = {
        'fc': fc_left,  # focal point
        'cc': cc_left,  # principle point
        'alpha_c': alpha_c_left,  # skew
        'kc': kc_left,  # distortion
    }

    intrinsic_right = {
        'fc': fc_right,
        'cc': cc_right,
        'kc': kc_right,
        'alpha_c': alpha_c_right,
    }

    intrin = intrinsic_left
    fc = intrin['fc'].ravel()
    cc = intrin['cc'].ravel()
    alpha_c = intrin['alpha_c'].ravel()[0]
    kc = intrin['kc'].ravel()
    KK = np.array([
        [fc[0], alpha_c * fc[0], cc[0]],
        [    0,           fc[1], cc[1]],
        [    0,               0,     1],
    ])


    CommandLine:
        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh

    """

    # Setup the input files
    import glob
    from os.path import expanduser, join

    data_fpath = expanduser('~/data/autoprocess_test_set')
    cal_fpath = join(data_fpath, 'cal_201608.mat')  # NOQA

    left_fpath = join(data_fpath, 'image_data/left')
    right_fpath = join(data_fpath, 'image_data/right')
    cam1_image_fpaths = sorted(glob.glob(join(left_fpath, '*.jpg')))
    cam2_image_fpaths = sorted(glob.glob(join(right_fpath, '*.jpg')))

    with open(join(data_fpath, 'cam1_images.txt'), 'w') as file:
        file.write('\n'.join(cam1_image_fpaths))

    with open(join(data_fpath, 'cam2_images.txt'), 'w') as file:
        file.write('\n'.join(cam2_image_fpaths))

    def add_stereo_camera_branch(pipe, prefix):
        """
        Helper that defines a single branch, so it can easilly be duplicated.
        """

        # ============
        # Branch Nodes
        # ============
        input_image = pipe.add_process(
            name=prefix + 'input_image', type='frame_list_input',
            config={
                'image_list_file': join(data_fpath, prefix + 'images.txt'),
                'frame_time': 0.03333333,
                'image_reader:type': 'ocv',
            })

        detect = pipe.add_process(
            name=prefix + 'detect', type='camtrawl_detect', config={ })

        detect.iports.connect({
            'image': input_image.oports['image']
        })

    pipe = sprokit_pipeline.Pipeline()
    add_stereo_camera_branch(pipe, 'cam1_')
    # add_stereo_camera_branch(pipe, 'cam2_')

    # pipe.add_process(name='measure', type='camtrawl_measure', config={})

    # pipe['measure'].iports.connect({
    #     # 'camera1': pipe['cam1_input_camera'].oports['camera'],
    #     # 'camera2': pipe['cam2_input_camera'].oports['camera'],
    #     'feature_set1': pipe['cam1_detect'].oports['feature_set'],
    #     'feature_set2': pipe['cam2_detect'].oports['feature_set'],
    # })

    pipe.config['_scheduler']['type'] = 'pythread_per_process'

    return pipe


def main():
    pipe = dummy_pipeline()
    # pipe = define_fishlen_pipeline()

    # Initially the main way to use the sprokit_pipeline module will be to
    # generate and dump the pipeline file to disk.
    pipe.write('camtrawl.pipe')

    pipe.draw_graph('pipeline.png')
    import utool as ut
    ut.startfile('pipeline.png')

    '''
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes
    export PYTHONPATH=$(pwd):$PYTHONPATH
    export SPROKIT_MODULE_PATH=$(pwd):$SPROKIT_MODULE_PATH
    export SPROKIT_PYTHON_MODULES=kwiver.processes:camtrawl_processes

    echo $SPROKIT_PYTHON_MODULES

    python camtrawl_pipeline_def.py
    ~/code/VIAME/build/install/bin/pipeline_runner -p camtrawl.pipe
    '''

    # Maybe sprokit_pipeline can have a convinience function to run a pipline
    # sprokit_pipeline.run('auto_fishlen.pipe')

    # Maybe the Pipeline can run itself as well?
    # pipe.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/VIAME/plugins/camtrawl/camtrawl_pipeline_def.py
    """
    main()
