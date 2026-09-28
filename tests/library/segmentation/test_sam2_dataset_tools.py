"""SAM2 dataset helpers preserve video colour and segmentation metrics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / 'packages/patches/sam2'


def load(relative):
    path = PATCHES / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_and_extraction_colour_conventions(tmp_path):
    from viame.video_io.frames import read_frames
    from PIL import Image
    clip = ROOT / 'tests/reference/inputs/clip.mp4'
    rgb = [frame for frame, _ in read_frames(str(clip))]
    visualizer = load('sav_dataset/utils/sav_utils.py')
    extractor = load('training/scripts/sav_frame_extraction_submitit.py')
    assert len(rgb) > 2
    np.testing.assert_array_equal(visualizer.decode_video(str(clip)), rgb)
    np.testing.assert_array_equal(extractor.decode_video(str(clip)), np.asarray(rgb)[..., ::-1])
    extractor.submitit_launch([str(clip)], 2, str(tmp_path))
    files = sorted((tmp_path / clip.stem).glob('*.jpg'))
    assert len(files) == len(rgb[::2])
    # Compare decoding to Pillow's own JPEG round trip: this catches swaps
    # without mistaking expected JPEG quantization for a decoder regression.
    reference = tmp_path / 'reference.jpg'
    Image.fromarray(rgb[0]).save(reference, quality=95)
    np.testing.assert_array_equal(np.asarray(Image.open(files[0])), np.asarray(Image.open(reference)))


def test_mask_visualization_keeps_hole_and_outer_borders():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    module = load('sav_dataset/utils/sav_utils.py')
    mask = np.zeros((30, 32), dtype=bool)
    mask[3:26, 3:28] = True
    mask[10:17, 10:19] = False
    figure, ax = plt.subplots()
    try:
        module.show_anns([mask], [(1., 0., 0.)])
        image = np.asarray(ax.images[-1].get_array())
        assert image[3, 7, 3] == 1
        assert image[9, 14, 3] == 1
        assert image[13, 14, 3] == 0
        np.testing.assert_allclose(image[6, 7], [1, 0, 0, .55])
    finally:
        plt.close(figure)


@pytest.mark.parametrize('boundary', [0, .008, .1])
def test_segmentation_boundary_metrics(boundary):
    module = load('sav_dataset/utils/sav_benchmark.py')
    with np.load(Path(__file__).with_name('sam2_boundary_reference.npz')) as data:
        for index in range(5):
            evaluator = module.Evaluator(boundary=boundary)
            evaluator.feed_frame(data[f'mask_{index}'], data[f'gt_{index}'])
            iou, boundary_f = evaluator.conclude()
            actual = np.array([list(iou.values()), list(boundary_f.values())])
            np.testing.assert_array_equal(actual, data[f'expected_{index}_{boundary}'])
