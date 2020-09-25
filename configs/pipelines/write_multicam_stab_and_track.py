from pathlib import Path
from textwrap import dedent

PIPELINE_DIR = Path(__file__).parent

python_scheduler = dedent("""\
    config _scheduler
      type = pythread_per_process

""")

def make_input_creator():
    default_input = (PIPELINE_DIR / 'common_default_input.pipe').read_text()
    default_input += '\n'
    def create_input(i):
        return (default_input
                .replace('process input', f'process input{i}')
                .replace('input_list.txt', f'input_list_{i}.txt'))
    return create_input

inadapt = 'process in_adapt :: input_adapter\n\n'

def make_detector_creator(embedded):
    model_dir = '../models' if embedded else 'models'
    def create_detector(i):
        return dedent(f"""\
            process detector{i}
              :: image_object_detector
              :detector:type            netharn

              block detector:netharn
                relativepath deployed = {model_dir}/sea_lion_multi_class.zip
              endblock

        """)
    return create_detector

def connect_input_detector(i):
    return dedent(f"""\
        connect from input{i}.image
                to detector{i}.image
    """)

def adapt_suffix(i):
    return str(i) if i > 1 else ''

def connect_inadapt_detector(i):
    return dedent(f"""\
        connect from in_adapt.image{adapt_suffix(i)}
                to detector{i}.image
    """)

def stabilize_images(ncam):
    return dedent(f"""\
        process stabilizer
          :: many_image_stabilizer
          n_input = {ncam}

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

def connect_input_stabilizer(i):
    return dedent(f"""\
        connect from input{i}.image
                to stabilizer.image{i}
    """)

def connect_inadapt_stabilizer(i):
    return dedent(f"""\
        connect from in_adapt.image{adapt_suffix(i)}
                to stabilizer.image{i}
    """)

def multitrack(ncam):
    return dedent(f"""\
        process tracker :: multicam_homog_tracker
          n_input = {ncam}

    """)

def connect_homog_tracker(i):
    return dedent(f"""\
        connect from stabilizer.homog{i}
                to tracker.homog{i}
    """)

def connect_det_tracker(i):
    return dedent(f"""\
        connect from detector{i}.detected_object_set
                to tracker.det_objs_{i}
    """)

connect_ts_tracker = dedent(f"""\
    connect from input1.timestamp
            to tracker.timestamp
""")

connect_ets_tracker = dedent(f"""\
    connect from in_adapt.timestamp
            to tracker.timestamp
""")


def write_tracks(i):
    return dedent(f"""\
        process write_tracks_{i} :: write_object_track
          file_name = tracks{i}.csv
          writer:type = viame_csv

    """)

def connect_track_write(i):
    return dedent(f"""\
        connect from tracker.obj_tracks_{i}
                to write_tracks_{i}.object_track_set
    """)

outadapt = 'process out_adapt :: output_adapter\n\n'

def connect_track_outadapt(i):
    return dedent(f"""\
        connect from tracker.obj_tracks_{i}
                to out_adapt.object_track_set{adapt_suffix(i)}
    """)

def connect_ts_outadapt(i):
    i = adapt_suffix(i)
    return dedent(f"""\
        connect from in_adapt.timestamp{i}
                to out_adapt.timestamp{i}
    """)

def connect_filename_outadapt(i):
    i = adapt_suffix(i)
    return dedent(f"""\
        connect from in_adapt.file_name{i}
                to out_adapt.file_name{i}
    """)

def write_homogs(i):
    return dedent(f"""\
        process write_homogs_{i}
          :: kw_write_homography
          output = homogs{i}.txt

    """)

def connect_stab_write(i):
    return dedent(f"""\
        connect from stabilizer.homog{i}
                to write_homogs_{i}.homography
    """)

def create_file(ncams, *, embedded):
    create_input = make_input_creator()
    create_detector = make_detector_creator(embedded)
    connect_in_stab = (connect_inadapt_stabilizer if embedded
                       else connect_input_stabilizer)
    connect_in_det = (connect_inadapt_detector if embedded
                      else connect_input_detector)
    connect_ts_track = (connect_ets_tracker if embedded
                        else connect_ts_tracker)
    rncams = range(1, ncams + 1)
    return ''.join([
        python_scheduler,
        *([inadapt] if embedded else map(create_input, rncams)),
        stabilize_images(ncams),
        *map(connect_in_stab, rncams), '\n',
        *map(create_detector, rncams),
        *map(connect_in_det, rncams), '\n',
        multitrack(ncams),
        *map(connect_homog_tracker, rncams), '\n',
        *map(connect_det_tracker, rncams), '\n',
        connect_ts_track, '\n',
        *([] if embedded else [
            *map(write_homogs, rncams),
            *map(connect_stab_write, rncams), '\n',
            *map(write_tracks, rncams),
            *map(connect_track_write, rncams),
        ]),
        *([
            outadapt,
            *map(connect_track_outadapt, rncams), '\n',
            *map(connect_ts_outadapt, rncams), '\n',
            *map(connect_filename_outadapt, rncams),
        ] if embedded else []),
         # '\n',
    ])

def main():
    for embedded in [False, True]:
        for ncams in [2, 3]:
            emb = 'embedded_dual_stream' if embedded else ''
            mst = f'tracker_sea_lion_{ncams}-cam.pipe'
            out = PIPELINE_DIR / emb / mst
            out.write_text(create_file(ncams, embedded=embedded))

if __name__ == '__main__':
    main()
