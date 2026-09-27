"""ORB, against a recording taken from cv2 5.0.0.

The image is built with VIAME's own blur so that the recording depends on
nothing outside this tree; the keypoints and descriptors below were checked
to be identical to `cv2.ORB_create( nfeatures=8, nlevels=3 )` on the same
bytes at the time they were taken. The wider verification -- 147
configurations over seven images, keypoints and descriptors both -- lives in
the commit message rather than here, since it needs cv2.
"""
import numpy as np
import pytest
from viame import image_kernels
from viame.image_processing import features


KEYPOINTS = [
    [96.0, 78.0, 31.0, 246.3669, 0.0012, 0.0],
    [107.0, 85.0, 31.0, 67.5672, 0.0022, 0.0],
    [107.0, 87.0, 31.0, 72.7635, 0.0008, 0.0],
    [93.6, 60.0, 37.2, 92.4713, 0.0001, 1.0],
    [100.8, 74.4, 37.2, 258.0101, 0.0002, 1.0],
    [96.0, 78.0, 37.2, 260.3905, 0.0012, 1.0],
    [93.6, 59.04, 44.64, 178.7125, 0.0001, 2.0],
    [100.8, 73.44, 44.64, 163.8996, 0.0002, 2.0],
]

FIRST_DESCRIPTOR = [
    57, 28, 83, 237, 20, 167, 105, 82, 85, 8, 242, 66, 51, 17, 49, 224,
    88, 168, 22, 168, 169, 186, 112, 159, 151, 236, 162, 0, 61, 137, 112, 40,
]

LAST_DESCRIPTOR = [
    37, 228, 104, 64, 76, 153, 229, 121, 90, 242, 116, 74, 16, 200, 214, 61,
    89, 40, 2, 46, 193, 246, 254, 94, 110, 26, 197, 44, 95, 184, 90, 164,
]


def make_image():
    rng = np.random.default_rng(303)
    image = image_kernels.gaussian_blur(
        rng.integers(0, 256, (120, 160), dtype=np.uint8), 5, 1.4)
    for _ in range(10):
        x = int(rng.integers(10, 140))
        y = int(rng.integers(10, 100))
        image[y:y + 14, x:x + 14] = int(rng.integers(0, 256))
    return image


def test_orb_reproduces_its_recording():
    keypoints, descriptors = features.orb(make_image(), n_features=8,
                                          n_levels=3)
    assert keypoints.shape == (8, 6)
    assert descriptors.shape == (8, 32)
    assert descriptors.dtype == np.uint8
    np.testing.assert_allclose(keypoints, KEYPOINTS, atol=5e-5)
    np.testing.assert_array_equal(descriptors[0], FIRST_DESCRIPTOR)
    np.testing.assert_array_equal(descriptors[-1], LAST_DESCRIPTOR)


def test_orb_returns_levels_in_order():
    # Detection order, which is by level and then by row and column within
    # one. cv2's order is whatever `std::nth_element` left behind, so this
    # is ours; orb.h says why nothing downstream can tell.
    keypoints, _ = features.orb(make_image(), n_features=40)
    levels = keypoints[:, 5]
    assert np.all(np.diff(levels) >= 0)
    assert keypoints[0, 5] == 0
    # The size is the patch scaled to the level it was found at.
    for level in np.unique(levels):
        sizes = keypoints[levels == level, 2]
        np.testing.assert_allclose(sizes, 31 * np.float32(1.2) ** level,
                                   rtol=1e-6)


def test_orb_describe_takes_the_detected_keypoints_back():
    image = make_image()
    keypoints, descriptors = features.orb(image, n_features=8, n_levels=3)
    again, described = features.orb_describe(image, keypoints, n_levels=3)
    np.testing.assert_array_equal(again, keypoints)
    np.testing.assert_array_equal(described, descriptors)


def test_orb_without_descriptors_still_detects():
    keypoints, descriptors = features.orb(make_image(), n_features=8,
                                          n_levels=3, describe=False)
    assert keypoints.shape == (8, 6)
    assert descriptors.shape == (0, 32)


def test_orb_refuses_what_it_has_not_implemented():
    image = make_image()
    # Both draw their sampling pattern from cv::RNG, so a descriptor built
    # from a different pattern is not comparable with cv2's at all.
    with pytest.raises(ValueError):
        features.orb(image, wta_k=3)
    with pytest.raises(ValueError):
        features.orb(image, patch_size=41)
    with pytest.raises(ValueError):
        features.orb(image, score_type='shi_tomasi')


def test_orb_on_an_image_with_nothing_to_find():
    keypoints, descriptors = features.orb(np.full((200, 200), 90, np.uint8))
    assert len(keypoints) == 0
    assert len(descriptors) == 0
