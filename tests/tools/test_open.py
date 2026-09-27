"""The public loader uses native containers, readers and shared recognition."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import viame
from viame import _io
from viame._file_formats import file_kind


def pixels(image):
    return image.image().asarray()


@pytest.fixture
def images(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / ("image %02d.png" % index)
        Image.fromarray(np.full((8, 12, 3), index * 70, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_image_and_header_detection(images):
    np.testing.assert_array_equal(pixels(viame.open(images[1])), 70)
    renamed = images[1].with_suffix('.data')
    renamed.write_bytes(images[1].read_bytes())
    assert file_kind(renamed) == "image"
    np.testing.assert_array_equal(pixels(viame.open(renamed)), 70)


def test_uint16_image(tmp_path):
    path = tmp_path / "depth.png"
    Image.fromarray(np.full((4, 5), 45000, np.uint16)).save(path)
    out = pixels(viame.open(path))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, 45000)


def test_relative_list_is_lazy_and_closeable(images, tmp_path, monkeypatch):
    path = tmp_path / 'images.txt'
    path.write_text('  # images\n\n' + '\n'.join(p.name for p in images))
    calls = []
    original = _io._load_image
    def load(reader, path):
        calls.append(path)
        return original(reader, path)
    monkeypatch.setattr(_io, '_load_image', load)
    with viame.open(path) as sequence:
        assert len(sequence) == 3
        assert calls == []
        first = next(sequence)
        assert len(calls) == 1
        assert sequence.timestamp.get_frame() == 1
        np.testing.assert_array_equal(pixels(first), 0)
    assert sequence.closed
    assert list(sequence) == []


def test_directory_and_list_have_same_order(images, tmp_path):
    path = tmp_path / 'images.txt'
    path.write_text('\n'.join(p.name for p in images))
    for sequence in (viame.open(path), viame.open(tmp_path)):
        assert [int(pixels(image).mean()) for image in sequence] == [0, 70, 140]
        assert sequence.closed


@pytest.mark.parametrize('rate', [0, -1, float('nan'), float('inf'), True])
def test_invalid_rates(images, rate):
    with pytest.raises(ValueError, match='frame_rate'):
        viame.open(images[0], rate)


def test_rate_rejected_for_images(images):
    with pytest.raises(ValueError, match='only for video'):
        viame.open(images[0], 5)


def test_invalid_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        viame.open(tmp_path / 'missing')
    for name, content in [('file.json', '{}'), ('file.csv', 'a,b,c\n'),
                          ('file.txt', ''), ('file.data', 'hello')]:
        path = tmp_path / name
        path.write_text(content)
        with pytest.raises(ValueError):
            viame.open(path)
    path = tmp_path / 'file.txt'
    path.write_text('missing.png\n')
    with pytest.raises(FileNotFoundError):
        viame.open(path)


@pytest.mark.parametrize('format_name', ['csv', 'dive', 'coco'])
def test_annotations_preserve_track_ids_and_states(tmp_path, format_name):
    path = tmp_path / ('tracks.csv' if format_name == 'csv' else 'tracks.json')
    if format_name == 'csv':
        path.write_text('7,frame.png,1,1,2,5,6,0.9,-1,fish,0.9\n'
                        '7,frame2.png,2,2,3,6,7,0.8,-1,fish,0.8\n')
    elif format_name == 'dive':
        path.write_text(json.dumps({'version': 2, 'tracks': {'7': {
            'id': 7, 'confidencePairs': [['fish', 0.9]], 'features': [
                {'frame': 1, 'bounds': [1, 2, 5, 6]},
                {'frame': 2, 'bounds': [2, 3, 6, 7]}]}}}))
    else:
        path.write_text(json.dumps({'images': [
            {'id': 1, 'file_name': 'frame.png', 'frame_index': 1},
            {'id': 2, 'file_name': 'frame2.png', 'frame_index': 2}],
            'categories': [{'id': 1, 'name': 'fish'}], 'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [1, 2, 4, 4], 'track_id': 7},
                {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [2, 3, 4, 4], 'track_id': 7}]}))
    tracks = viame.open(path).tracks()
    assert len(tracks) == 1
    assert tracks[0].id == 7
    assert len(tracks[0]) == 2
    states = list(tracks[0])
    assert [s.frame_id for s in states] == [1, 2]
    assert states[0].detection().bounding_box.min_x() == 1


@pytest.mark.parametrize('doc', [{'tracks': {}}, {'images': [], 'annotations': []}])
def test_empty_annotations(tmp_path, doc):
    path = tmp_path / 'empty.json'
    path.write_text(json.dumps(doc))
    assert len(viame.open(path).tracks()) == 0


@pytest.fixture
def video(tmp_path):
    import shutil
    import subprocess
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg needed to generate the integration fixture')
    path = tmp_path / 'clip.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x48:rate=10:duration=2', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


@pytest.mark.parametrize('keyword', [False, True])
def test_video_sampling(video, keyword):
    sequence = viame.open(video, frame_rate=5) if keyword else viame.open(video, 5)
    with sequence:
        frames, times = [], []
        for image in sequence:
            assert pixels(image).shape == (48, 64, 3)
            frames.append(sequence.timestamp.get_frame())
            times.append(sequence.timestamp.get_time_seconds())
    assert frames == list(range(1, 21, 2))
    np.testing.assert_allclose(times, np.arange(10) / 5, atol=1e-6)
    assert sequence.closed


def test_video_without_sampling_and_early_close(video):
    assert len(list(viame.open(video))) == 20
    with viame.open(video, 100) as sequence:
        assert next(sequence) is not None
    assert sequence.closed
    assert list(sequence) == []


def test_vfr_sampling_uses_time_and_releases_reader(monkeypatch, tmp_path):
    class Stamp:
        def __init__(self, seconds=None): self.seconds = seconds
        def has_valid_time(self): return self.seconds is not None
        def get_time_seconds(self): return self.seconds
    class Reader:
        closed = False
        def open(self, path): self.index = -1
        def frame_rate(self): return 30
        def next_frame(self):
            self.index += 1
            return self.index < 6
        def frame_timestamp(self): return Stamp([3, 3.03, 3.11, 3.22, 3.39, 3.45][self.index])
        def frame_image(self): return self.index
        def close(self): self.closed = True
    reader = Reader()
    monkeypatch.setattr(_io, '_algorithms', lambda: (
        SimpleNamespace(VideoInput=object), SimpleNamespace(Timestamp=Stamp)))
    monkeypatch.setattr(_io, '_create', lambda *args: reader)
    assert list(_io.VideoSequence('unused', 5)) == [0, 3, 5]
    assert reader.closed


def test_sequence_closes_on_decode_error(images, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('decode failed')
    sequence = viame.open(tmp_path)
    monkeypatch.setattr(_io, '_load_image', fail)
    with pytest.raises(OSError, match='decode failed'):
        next(sequence)
    assert sequence.closed


def test_inspect_and_open_share_recognition(images, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('open_test_inspect',
        Path(__file__).resolve().parents[2] / 'tools' / 'inspect_file.py')
    inspect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect)
    assert inspect.file_kind is _io.file_kind
    assert inspect.json_annotation_format is _io.json_annotation_format
    assert inspect.image_list_entries is _io.image_list_entries
    assert inspect.inspect_path(str(images[0])).category == 'image'
    path = tmp_path / 'images.txt'
    path.write_text('  # comment\n' + images[0].name)
    assert inspect.inspect_path(str(path)).integrity.startswith('ok')


@pytest.mark.parametrize('source_rate,expected', [(10, [0, 2, 4]), (0, None)])
def test_sampling_without_timestamps(monkeypatch, source_rate, expected):
    class Stamp:
        def has_valid_time(self): return False
    class Reader:
        closed = False
        def open(self, path): self.index = -1
        def frame_rate(self): return source_rate
        def next_frame(self):
            self.index += 1
            return self.index < 5
        def frame_timestamp(self): return Stamp()
        def frame_image(self): return self.index
        def close(self): self.closed = True
    reader = Reader()
    monkeypatch.setattr(_io, '_algorithms', lambda: (
        SimpleNamespace(VideoInput=object), SimpleNamespace(Timestamp=Stamp)))
    monkeypatch.setattr(_io, '_create', lambda *args: reader)
    if expected is None:
        with pytest.raises(ValueError, match='neither timestamps'):
            list(_io.VideoSequence('unused', 5))
    else:
        assert list(_io.VideoSequence('unused', 5)) == expected
    assert reader.closed


@pytest.fixture
def pipeline_templates(monkeypatch):
    # Use this checkout's templates, independent of a stale installation.
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'configs' / 'pipelines'
    monkeypatch.setattr(_io, '_pipeline_root', lambda: root)
    return root


def test_open_pipe_and_run(tmp_path, monkeypatch):
    import subprocess
    path = tmp_path / 'a space.pipe'
    path.write_text('config global\n  value = original\n')
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(_io.subprocess, 'run', run)
    with viame.open(path) as pipe:
        assert isinstance(pipe, viame.Pipeline)
        assert pipe.path == str(path)
        assert calls == []  # Opening does not execute the pipeline.
        result = pipe.run(tmp_path / 'input file.mp4', output_dir=tmp_path / 'out',
                          frame_rate=5, args=['--no-reset-prompt'], capture_output=True)
        assert result.returncode == 0
        assert calls[-1][0][1:] == ['run', str(path), str(tmp_path / 'input file.mp4'),
                                    '-o', str(tmp_path / 'out'), '-frate', '5',
                                    '--no-reset-prompt']
        assert calls[-1][1] == dict(check=True, text=True, capture_output=True)
        pipe.run(args=['--dump-pipe'], check=False)
        assert calls[-1][0][1:] == ['run', str(path), '--dump-pipe']
        assert calls[-1][1]['check'] is False
        with pytest.raises(TypeError):
            pipe.run(args='--dump-pipe')
        with pytest.raises(ValueError):
            pipe.run(frame_rate=5)
        for rate in (True, 0, -1, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                pipe.run('video.mp4', frame_rate=rate)
    assert path.exists()
    pipe.close()
    with pytest.raises(ValueError):
        pipe.run()
    with pytest.raises(ValueError):
        pipe.__enter__()


def test_open_pipe_native_runner(tmp_path):
    path = tmp_path / 'a space.pipe'
    path.write_text('config global\n  value = original\n')
    with viame.open(path) as pipe:
        result = pipe.run(args=['--dump-pipe'], capture_output=True, timeout=60)
    assert 'original' in result.stdout


def test_open_pipeline_zip_selection_and_cleanup(tmp_path, pipeline_templates):
    from pathlib import Path
    import zipfile
    archive = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('nested/a.pipe', 'process detector_input\n  :: image_filter\n')
        zf.writestr('nested/b.pipe', 'include common_default_input.pipe\n')
        zf.writestr('nested/weights.pt', b'weights')
    with pytest.raises(ValueError, match='several pipelines'):
        viame.open(archive)
    with pytest.raises(ValueError, match='Unknown pipeline'):
        viame.open(archive, pipeline='missing.pipe')
    with viame.open(archive, pipeline='nested/a.pipe') as pipe:
        work = Path(pipe._work.name)
        text = Path(pipe.path).read_text()
        assert 'to   detector_input.image' in text
        assert (work / 'bundle/nested/weights.pt').read_bytes() == b'weights'
        assert str(work / 'bundle/nested/a.pipe') in text
    assert not work.exists()
    with viame.open(archive, pipeline='nested/b.pipe') as pipe:
        assert pipe.path.endswith('/bundle/nested/b.pipe')
        assert Path(pipe.path).read_text() == 'include common_default_input.pipe\n'


@pytest.mark.parametrize('kind', ['onnx', 'onnx_zip', 'darknet_zip', 'torch', 'torch_zip'])
def test_open_models_share_run_wrappers(tmp_path, pipeline_templates, kind):
    from pathlib import Path
    import pickle
    import zipfile
    if kind == 'onnx':
        path = tmp_path / 'model.onnx'
        path.write_bytes(b'')
    elif kind == 'onnx_zip':
        path = tmp_path / 'model.zip'
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr('model.onnx', b'')
    elif kind == 'darknet_zip':
        path = tmp_path / 'model.zip'
        with zipfile.ZipFile(path, 'w') as zf:
            for name in ('model.weights', 'model.cfg', 'model.lbl'):
                zf.writestr(name, b'')
    else:
        path = tmp_path / 'model.pth'
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr('archive/data.pkl', pickle.dumps({
                'model': {'transformer.decoder.w': 0, 'class_embed.weight': 0}}, protocol=2))
        if kind == 'torch_zip':
            outer = tmp_path / 'model.zip'
            with zipfile.ZipFile(outer, 'w') as zf:
                zf.write(path, 'model.pth')
            path = outer
    with viame.open(path) as pipe:
        assert pipe.info.runnable
        expected_impl = 'onnx' if kind.startswith('onnx') else 'darknet' if kind == 'darknet_zip' else 'rf_detr'
        assert pipe.info.impl == expected_impl
        assert expected_impl in Path(pipe.path).read_text()
        work = Path(pipe._work.name)
    assert not work.exists()


def test_open_bad_archive_cleans_up(tmp_path, monkeypatch):
    from pathlib import Path
    import zipfile
    created = []
    original = _io.tempfile.TemporaryDirectory
    def temporary(*args, **kwargs):
        temp = original(*args, **kwargs)
        created.append(Path(temp.name))
        return temp
    monkeypatch.setattr(_io.tempfile, 'TemporaryDirectory', temporary)
    path = tmp_path / 'bad.zip'
    path.write_bytes(b'not a zip')
    with pytest.raises(ValueError, match='unreadable archive'):
        viame.open(path)
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('readme.txt', 'No model')
    with pytest.raises(ValueError, match='unrecognized archive'):
        viame.open(path)
    assert created and all(not p.exists() for p in created)


def test_pipeline_options_rejected_for_data(images):
    with pytest.raises(ValueError, match='pipeline'):
        viame.open(images[0], pipeline='a.pipe')


def test_open_zipped_pipeline_processes_image(tmp_path, pipeline_templates):
    import zipfile
    # A native detector exercises extraction, include resolution, the run
    # applet's input dispatch and its output writer without model downloads.
    archive = tmp_path / 'detector.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.write(pipeline_templates / 'detector_simple_hough.pipe', 'detector.pipe')
        for name in ('common_default_input_with_downsampler.pipe', 'common_default_input.pipe'):
            zf.write(pipeline_templates / name, name)
    image = tmp_path / 'image.png'
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image)
    with viame.open(archive, pipeline='detector.pipe') as detector:
        result = detector.run(image, output_dir=tmp_path / 'results',
                              args=['--no-reset-prompt'], capture_output=True, timeout=60, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((tmp_path / 'results').glob('*detections.csv'))
    assert len(outputs) == 1, result.stdout + result.stderr
    assert '# 1: Detection or Track-id' in outputs[0].read_text()


def _memory_pipe(tmp_path, cameras=1, reader='video_input'):
    path = tmp_path / 'memory.pipe'
    lines = ['process result\n :: output_adapter\n']
    for index in range(cameras):
        name = 'input' if cameras == 1 else 'input{}'.format(index + 1)
        lines.append('process {}\n :: {}\n :video_filename /missing/images.txt\n'.format(name, reader))
        for field in ('image', 'timestamp', 'file_name', 'frame_rate'):
            lines.append('connect from {0}.{1} to result.{0}_{1}\n'.format(name, field))
    path.write_text('\n'.join(lines))
    return path


@pytest.mark.parametrize('cameras', [1, 2, 3])
def test_embedded_memory_inputs(tmp_path, cameras):
    path = _memory_pipe(tmp_path, cameras)
    with viame.open(path, embedded=True) as pipeline:
        assert isinstance(pipeline, viame.EmbeddedPipeline)
        for frame in range(4):
            images = {name: np.full((8, 12, 3), frame + index, np.uint8)
                      for index, name in enumerate(pipeline.input_names)}
            pipeline.send(next(iter(images.values())) if cameras == 1 else images,
                          frame_rate=10)
            output = pipeline.receive()
            for name, image in images.items():
                np.testing.assert_array_equal(pixels(output['result.' + name + '_image']), image)
                stamp = output['result.' + name + '_timestamp']
                assert stamp.get_frame() == frame + 1
                assert stamp.get_time_seconds() == pytest.approx(frame / 10)
                assert output['result.' + name + '_frame_rate'] == 10
        prepared = pipeline.path
    from pathlib import Path
    assert pipeline.closed and not Path(prepared).exists()
    assert path.exists()
    pipeline.close()
    with pytest.raises(ValueError, match='closed'):
        pipeline.send(images)
    with pytest.raises(ValueError, match='closed'):
        pipeline.receive()


def test_embedded_close_drains_pending_results(tmp_path):
    # More outputs than the native output queue holds. close must drain while
    # sending EOF and waiting, including when none of the results were read.
    with viame.open(_memory_pipe(tmp_path), embedded=True) as pipeline:
        for _ in range(6):
            pipeline.send(np.zeros((8, 8, 3), np.uint8))
    with viame.open(_memory_pipe(tmp_path), embedded=True):
        pass  # Closing without sending any frames must also terminate.


def test_embedded_validation_and_metadata(images, tmp_path):
    with pytest.raises(ValueError, match='only for pipelines'):
        viame.open(images[0], embedded=True)
    path = _memory_pipe(tmp_path, 2)
    with pytest.raises(ValueError, match='require embedded'):
        viame.open(path, inputs=['input1'])
    with pytest.raises(ValueError, match='Select inputs'):
        viame.open(path, embedded=True, inputs=['missing'])
    with viame.open(path, embedded=True) as pipeline:
        image = viame.open(images[0])
        with pytest.raises(TypeError, match='send'):
            pipeline.run('video.mp4')
        with pytest.raises(ValueError, match='by input process'):
            pipeline.send(image)
        with pytest.raises(ValueError, match='Unknown image'):
            pipeline.send({'typo': image})
        with pytest.raises(ValueError, match='Supply reader port'):
            pipeline.send({'input1': image})
        with pytest.raises(ValueError, match='Unknown input ports'):
            pipeline.send({}, values={'typo': 1})
        for rate in (True, 0, -1, float('inf'), float('nan')):
            with pytest.raises(ValueError, match='frame_rate'):
                pipeline.send({}, frame_rate=rate)
        _, types = _io._algorithms()
        stamp = types.Timestamp()
        stamp.set_frame(42)
        stamp.set_time_seconds(7)
        pipeline.send({'input1': image, 'input2': image}, timestamp=stamp,
                      values={'input1.file_name': 'left.png'})
        out = pipeline.receive()
        assert out['result.input1_file_name'] == 'left.png'
        assert out['result.input2_timestamp'].get_frame() == 42


def test_embedded_native_detector_and_zip(tmp_path, pipeline_templates, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import zipfile
    archive = tmp_path / 'detector.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.write(pipeline_templates / 'detector_simple_hough.pipe', 'detector.pipe')
        for name in ('common_default_input_with_downsampler.pipe', 'common_default_input.pipe'):
            zf.write(pipeline_templates / name, name)
    with viame.open(archive, pipeline='detector.pipe', embedded=True) as detector:
        assert detector.input_names == ('input',)
        for _ in range(3):
            detector.send(np.zeros((64, 64, 3), np.uint8))
            output = detector.receive()
            assert len(output['detector_writer.detected_object_set']) == 0
    assert not (tmp_path / 'computed_detections.csv').exists()


def test_embedded_parser_preserves_included_relative_paths(tmp_path):
    from viame._embedded import _prepare
    folder = tmp_path / 'sub'
    folder.mkdir()
    (folder / 'input.pipe').write_text('''process input
 :: video_input
 :video_filename missing.txt
process detector
 :: image_filter
 relativepath filter:model = weights.bin
''')
    pipe = tmp_path / 'main.pipe'
    pipe.write_text('''include sub/input.pipe
process writer
 :: image_writer
connect from input.image to detector.image
connect from detector.image to writer.image
''')
    description = _prepare(pipe, None, None)
    text, names, outputs = (description.pipeline_text, tuple(description.input_names),
                            description.output_ports)
    assert str(folder / 'weights.bin') in text
    assert 'video_input' not in text and 'image_writer' not in text
    assert names == ('input',)
    assert 'writer.image' in outputs


def test_embedded_explicit_process_selection(tmp_path):
    # Selecting custom source/sink names does not instantiate their types.
    path = _memory_pipe(tmp_path, reader='custom_source')
    path.write_text(path.read_text().replace('output_adapter', 'custom_sink'))
    with viame.open(path, embedded=True, inputs='input', outputs='result') as pipe:
        pipe.send(np.zeros((4, 5, 3), np.uint8))
        assert pixels(pipe.receive()['result.input_image']).shape == (4, 5, 3)


def test_embedded_sampling_and_receive_timeout(tmp_path, pipeline_templates):
    path = tmp_path / 'sampling.pipe'
    path.write_text('''include {}/common_default_input_with_downsampler.pipe
process result
 :: output_adapter
connect from downsampler.output_1 to result.image
'''.format(pipeline_templates))
    with viame.open(path, embedded=True) as pipe:
        with pytest.raises(TimeoutError):
            pipe.receive(timeout=0)
        with pytest.raises(TypeError, match='Images must'):
            pipe.send(object())
        pipe.send(np.zeros((8, 8, 3), np.uint8), frame_rate=30)
        assert 'result.image' in pipe.receive(timeout=10)
        pipe.send(np.zeros((8, 8, 3), np.uint8), frame_rate=30)
        with pytest.raises(TimeoutError):
            pipe.receive(timeout=0.05)  # 5 Hz pipeline drops this 30 Hz frame.
        for rate in (-1, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='timeout'):
                pipe.receive(timeout=rate)


@pytest.mark.parametrize('entry_point', ['create', 'embedded'])
def test_automatic_registration_in_fresh_python_process(entry_point, tmp_path):
    # Other tests initialize plugins in their interpreter. Use a fresh process
    # to prove the lower-level entry points work without viame.open or a manual
    # load_known_modules call first.
    import subprocess
    import sys
    import textwrap
    source = '''
from kwiver.vital import algo
from kwiver.vital.modules import modules
from kwiver.sprokit.adapters import adapter_data_set, embedded_pipeline
from pathlib import Path
import sys

if sys.argv[1] == 'create':
    assert algo.ImageIO.create('ocv') is not None
    assert algo.ReadObjectTrackSet.create('viame_csv') is not None
else:
    path = Path(sys.argv[2]) / 'automatic.pipe'
    path.write_text('process input\\n :: input_adapter\\n'
                    'process output\\n :: output_adapter\\n'
                    'connect from input.value to output.value\\n')
    pipeline = embedded_pipeline.EmbeddedPipeline()
    pipeline.build_pipeline(str(path), str(path.parent))
    pipeline.start()
    data = adapter_data_set.AdapterDataSet.create()
    data['value'] = 42
    pipeline.send(data)
    assert pipeline.receive()['value'] == 42
    pipeline.send_end_of_input()
    assert pipeline.receive().is_end_of_data()
    pipeline.wait()

before = list(algo.ReadObjectTrackSet.registered_names())
assert 'viame_csv' in before
for _ in range(3):
    modules.load_known_modules()  # Explicit initialization remains supported.
    assert algo.ImageIO.create('ocv') is not None
    assert algo.ReadObjectTrackSet.create('viame_csv') is not None
    assert list(algo.ReadObjectTrackSet.registered_names()) == before
'''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source),
                             entry_point, str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
