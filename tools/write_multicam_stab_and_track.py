# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from pathlib import Path
from textwrap import dedent

import os

PIPELINE_DIR = Path(__file__).parent / '../configs/pipelines'

def enum1(it): return enumerate(it, 1)
def range1(x): return range(1, x + 1)

python_scheduler = dedent("""\
    config _scheduler
      type = pythread_per_process

""")

def make_input_creator():
    default_input = (PIPELINE_DIR / 'common_default_input.pipe').read_text()
    include = '\ninclude common_default_input.pipe\n'
    downsampler = PIPELINE_DIR / 'common_default_input_with_downsampler.pipe'
    downsampler = downsampler.read_text().replace(include, '')
    template = default_input + os.linesep + downsampler + os.linesep
    def create_input(i):
        text = (template
                .replace('process input', f'process input{i}')
                .replace('input_list.txt', f'input_list_{i}.txt')
                .replace('process downsampler', f'process downsampler{i}')
                .replace('from input', f'from input{i}')
                .replace('to   downsampler', f'to   downsampler{i}'))
        # Ports are image, file name, and timestamp
        ports = 'output_1', 'output_2', 'timestamp'
        ports = [f'downsampler{i}.{p}' for p in ports]
        return text, ports
    return create_input

def inadapt(ncams):
    process = 'process in_adapt :: input_adapter\n\n'
    return process, [[
        f'in_adapt.{prefix}{adapt_suffix(i)}' for i in range1(ncams)
    ] for prefix in ['image', 'file_name', 'timestamp']]

def make_detector_creator(embedded):
    model_dir = '../models' if embedded else 'models'
    def create_detector(i, image_port):
        return dedent(f"""\
            process detector{i}
              :: image_object_detector
              :detector:type            netharn

              block detector:netharn
                relativepath deployed = {model_dir}/sea_lion_v2_cfrnn_all_classes.zip
              endblock

            connect from {image_port}
                    to detector{i}.image

        """), f'detector{i}.detected_object_set'
    return create_detector

def adapt_suffix(i):
    return str(i) if i > 1 else ''

def stabilize_images(image_ports):
    process = dedent(f"""\
        process stabilizer
          :: many_image_stabilizer
          n_input = {len(image_ports)}

          feature_detector:type = filtered
          block feature_detector:filtered
            detector:type = ocv_SURF
            block detector:ocv_SURF
              extended           = false
              hessian_threshold  = 400
              n_octave_layers    = 3
              n_octaves          = 4
              upright            = false
            endblock

            filter:type = nonmax
            block filter:nonmax
              num_features_target = 5000
              num_features_range = 500
            endblock
          endblock

          descriptor_extractor:type = ocv_SURF
          block descriptor_extractor:ocv_SURF
            extended           = false
            hessian_threshold  = 400 # 5000
            n_octave_layers    = 3
            n_octaves          = 4
            upright            = false
          endblock

          feature_matcher:type = ocv_flann_based

          homography_estimator:type = vxl

          ref_homography_computer:type = core
          block ref_homography_computer:core
            backproject_threshold = 4
            allow_ref_frame_regression = false
            min_matches_threshold = 50
            estimator:type = vxl
            forget_track_threshold = 5
            inlier_scale = 10
            min_track_length = 1
            use_backproject_error = false
          endblock

    """)
    conns = ''.join(dedent(f"""\
        connect from {ip}
                to stabilizer.image{i}
    """) for i, ip in enum1(image_ports))
    out_ports = [f'stabilizer.homog{i}' for i in range1(len(image_ports))]
    return process + conns + '\n', out_ports

def multitrack(homogs, objects, timestamp):
    ncam, = {len(homogs), len(objects)}
    result = []
    result.append(dedent(f"""\
        process tracker :: multicam_homog_tracker
          n_input = {ncam}

    """))
    for i, h in enum1(homogs):
        result.append(dedent(f"""\
            connect from {h}
                    to tracker.homog{i}
        """))
    result.append('\n')
    for i, o in enum1(objects):
        result.append(dedent(f"""\
            connect from {o}
                    to tracker.det_objs_{i}
        """))
    result.append(dedent(f"""\

        connect from {timestamp}
                to tracker.timestamp

    """))
    return ''.join(result), [f'tracker.obj_tracks_{i}' for i in range1(ncam)]

def suppressor(homogs, objects, images):
    ncams, = {len(homogs), len(objects), len(images)}
    result = []
    result.append(dedent(f"""\
        process suppressor :: multicam_homog_det_suppressor
          n_input = {ncams}
          suppression_poly_class = Suppressed

    """))
    for prefix, ports in [('homog', homogs),
                          ('det_objs_', objects),
                          ('image', images)]:
        for i, p in enum1(ports):
            result.append(dedent(f"""\
                connect from {p}
                        to suppressor.{prefix}{i}
            """))
        result.append('\n')
    return ''.join(result), [f'suppressor.det_objs_{i}' for i in range1(ncams)]

def make_track_initializer():
    template = (PIPELINE_DIR / 'common_default_initializer.pipe').read_text()
    template += '\n'
    def initialize_tracks(i, object_port, timestamp):
        name = f'track_initializer_{i}'
        process = template.replace('track_initializer\n', f'{name}\n')
        return process + dedent(f"""\
            connect from {object_port}
                    to {name}.detected_object_set
            connect from {timestamp}
                    to {name}.timestamp

        """), f'{name}.object_track_set'
    return initialize_tracks

def write_tracks(i, track_port, timestamp, file_name):
    proc = f'track_writer{i}'
    return dedent(f"""\
        process {proc} :: write_object_track
          file_name = tracks{i}.csv
          frame_list_output = track_images_{i}.txt
          writer:type = viame_csv

        connect from {track_port}
                to {proc}.object_track_set
        connect from {timestamp}
                to {proc}.timestamp
        connect from {file_name}
                to {proc}.image_file_name

    """), None

def write_dets(i, det_port, file_name):
    proc = f'detector_writer{i}'
    return dedent(f"""\
        process {proc} :: detected_object_output
          file_name = detections{i}.csv
          frame_list_output = det_images_{i}.txt
          writer:type = viame_csv

        connect from {det_port}
                to {proc}.detected_object_set
        connect from {file_name}
                to {proc}.image_file_name

    """), None

def outadapt(track_sets, timestamps, file_names):
    result = []
    result.append('process out_adapt :: output_adapter\n\n')
    for prefix, ports in [('object_track_set', track_sets),
                          ('timestamp', timestamps),
                          ('file_name', file_names)]:
        for i, p in enum1(ports):
            result.append(dedent(f"""\
                connect from {p}
                        to out_adapt.{prefix}{adapt_suffix(i)}
            """))
        result.append('\n')
    return ''.join(result), None

def write_homogs(i, homog_port):
    return dedent(f"""\
        process homog_writer{i}
          :: kw_write_homography
          output = homogs{i}.txt

        connect from {homog_port}
                to homog_writer{i}.homography

     """), None

def create_file(type_, ncams, *, embedded):
    result = []
    append = result.append
    def do(x):
        text, val = x
        append(text)
        return val
    rncams = range1(ncams)
    append(python_scheduler)
    if embedded:
        images, file_names, timestamps = do(inadapt(ncams))
    else:
        create_input = make_input_creator()
        ports = (do(create_input(i)) for i in rncams)
        images, file_names, timestamps = zip(*ports)
    homogs = do(stabilize_images(images))
    create_detector = make_detector_creator(embedded)
    objects = [do(create_detector(i, im)) for i, im in enum1(images)]
    if type_ == 'tracker':
        track_sets = do(multitrack(homogs, objects, timestamps[0]))
        for i, ts, t, fn in zip(rncams, track_sets, timestamps, file_names):
            do(write_tracks(i, ts, t, fn))
    elif type_ == 'suppressor':
        objects_supp = do(suppressor(homogs, objects, images))
        for i, o, fn in zip(rncams, objects_supp, file_names):
            do(write_dets(i, o, fn))
        if embedded:
            initialize_tracks = make_track_initializer()
            track_sets = [do(initialize_tracks(i, o, t)) for i, o, t
                          in zip(rncams, objects_supp, timestamps)]
    else:
        raise ValueError
    for i, h in enum1(homogs):
        do(write_homogs(i, h))
    if embedded:
        do(outadapt(track_sets, timestamps, file_names))
    return ''.join(result).rstrip('\n') + '\n'

def main():
    for type_ in ['tracker', 'suppressor']:
        for embedded in [False, True]:
            for ncams in range(1, 4):
                emb = 'embedded_dual_stream' if embedded else ''
                mst = f'{type_}_sea_lion_{ncams}-cam.pipe'
                out = PIPELINE_DIR / emb / mst
                out.write_text(create_file(type_, ncams, embedded=embedded))

if __name__ == '__main__':
    main()
