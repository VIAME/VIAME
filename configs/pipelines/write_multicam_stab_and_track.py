from pathlib import Path
from textwrap import dedent

PIPELINE_DIR = Path(__file__).parent

def enum1(it): return enumerate(it, 1)
def range1(x): return range(1, x + 1)

python_scheduler = dedent("""\
    config _scheduler
      type = pythread_per_process

""")

def make_input_creator():
    default_input = (PIPELINE_DIR / 'common_default_input.pipe').read_text()
    default_input += '\n'
    def create_input(i):
        text = (default_input
                .replace('process input', f'process input{i}')
                .replace('input_list.txt', f'input_list_{i}.txt'))
        ports = [f'input{i}.{p}' for p in ['image', 'file_name', 'timestamp']]
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
                relativepath deployed = {model_dir}/sea_lion_multi_class.zip
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

def write_tracks(i, track_port):
    return dedent(f"""\
        process write_tracks_{i} :: write_object_track
          file_name = tracks{i}.csv
          writer:type = viame_csv

        connect from {track_port}
                to write_tracks_{i}.object_track_set

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
        process write_homogs_{i}
          :: kw_write_homography
          output = homogs{i}.txt

        connect from {homog_port}
                to write_homogs_{i}.homography

     """), None

def create_file(ncams, *, embedded):
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
    track_sets = do(multitrack(homogs, objects, timestamps[0]))
    for i, h in enum1(homogs):
        do(write_homogs(i, h))
    for i, ts in enum1(track_sets):
        do(write_tracks(i, ts))
    if embedded:
        do(outadapt(track_sets, timestamps, file_names))
    return ''.join(result).rstrip('\n') + '\n'

def main():
    for embedded in [False, True]:
        for ncams in [2, 3]:
            emb = 'embedded_dual_stream' if embedded else ''
            mst = f'tracker_sea_lion_{ncams}-cam.pipe'
            out = PIPELINE_DIR / emb / mst
            out.write_text(create_file(ncams, embedded=embedded))

if __name__ == '__main__':
    main()
