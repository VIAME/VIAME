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
