"""GrabCut against a recording taken from cv2 5.0.0.

The scene is built arithmetically so the recording depends on nothing outside
this tree, and the mask below was checked identical to `cv2.grabCut` on the
same bytes. The wider verification -- 300 configurations over five scene
kinds, four sizes, three generator states, one to three passes and both init
modes, masks **and** the fitted mixture models -- needs cv2 and lives in the
commit message.

Run just these:  ctest -R "unit:image_kernels:grabcut"
"""
import numpy as np
import pytest
from viame.image_kernels import grab_cut


def scene():
    """A patterned background with a two-tone block in the middle."""
    rows, columns = np.indices((16, 20))
    image = np.stack([(columns * 7 + 20) % 200 + 30,
                      (rows * 11 + 40) % 200 + 30,
                      (columns * 5 + rows * 3) % 200 + 30],
                     -1).astype(np.uint8)
    image[4:12, 5:15] = [210, 60, 70]
    image[5:11, 6:14] = [225, 45, 55]
    return np.ascontiguousarray(image)


RECT = [4, 3, 12, 10]

# 0 background, 2 probably background, 3 probably foreground. The block comes
# back as foreground with the one-pixel frame of the rectangle left probably
# background, which is what a correct cut of this scene looks like.
EXPECTED = np.array(
    [[0] * 20,
     [0] * 20,
     [0] * 20,
     [0] * 4 + [2] * 12 + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] + [3] * 10 + [2] + [0] * 4,
     [0] * 4 + [2] * 12 + [0] * 4,
     [0] * 20,
     [0] * 20,
     [0] * 20], dtype=np.uint8)


def test_grab_cut_reproduces_its_recording():
    mask, background, foreground = grab_cut(
        scene(), np.zeros((16, 20), np.uint8), rect=RECT, iterations=2,
        mode="rect")

    np.testing.assert_array_equal(mask, EXPECTED)
    assert background.shape == (65,)
    assert foreground.shape == (65,)
    # The five component weights sum to one on each side, and the foreground
    # has collapsed onto two components -- there are only two colours in the
    # block, so three of the five find nothing and are switched off.
    np.testing.assert_allclose(background[:5].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(foreground[:5].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(sorted(foreground[:5]),
                               [0.0, 0.0, 0.0, 0.4, 0.6], atol=1e-12)


def test_grab_cut_does_not_move_a_certain_label():
    # What makes a definite mark a constraint rather than a hint: the two
    # certain labels come back exactly as they went in, whatever the cut says.
    image = scene()
    mask = np.full((16, 20), 2, np.uint8)
    mask[0, :] = 0
    mask[15, :] = 1
    mask[6:10, 7:13] = 1

    out, _, _ = grab_cut(image, mask, iterations=2, mode="mask")

    assert np.all(out[0, :] == 0)
    assert np.all(out[15, :] == 1)
    assert np.all(out[6:10, 7:13] == 1)


def test_grab_cut_writes_the_mask_in_place_except_from_a_rectangle():
    image = scene()

    given = np.full((16, 20), 2, np.uint8)
    given[0, :] = 0
    given[5:11, 6:14] = 3
    out, _, _ = grab_cut(image, given, iterations=1, mode="mask")
    np.testing.assert_array_equal(out, given)

    # `rect` replaces it, so the array handed in is left alone.
    untouched = np.zeros((16, 20), np.uint8)
    out, _, _ = grab_cut(image, untouched, rect=RECT, iterations=1,
                         mode="rect")
    assert np.any(out != untouched)


def test_grab_cut_carries_its_models_between_calls():
    image = scene()
    first, background, foreground = grab_cut(
        image, np.zeros((16, 20), np.uint8), rect=RECT, iterations=1,
        mode="rect")

    # `eval` refits nothing at the start and uses the models it is given, so
    # handing back the pair from the first call continues it.
    again, _, _ = grab_cut(image, first.copy(), iterations=1, mode="eval",
                           background_model=background,
                           foreground_model=foreground)
    assert again.shape == first.shape

    # `eval_frozen` does one pass and leaves the models alone.
    frozen, b2, f2 = grab_cut(image, first.copy(), iterations=5,
                              mode="eval_frozen",
                              background_model=background,
                              foreground_model=foreground)
    np.testing.assert_array_equal(b2, background)
    np.testing.assert_array_equal(f2, foreground)


def test_grab_cut_is_a_function_of_its_input():
    # cv2's is not: its k-means seeding draws from `cv::theRNG()`, a global
    # mutable generator, so a call's answer depends on what else in the
    # process drew from it first -- seeding it differently moved 28 percent
    # of one 90 by 120 mask. This starts from the state a fresh process has,
    # every time.
    image = scene()
    runs = [grab_cut(image, np.zeros((16, 20), np.uint8), rect=RECT,
                     iterations=2, mode="rect")[0] for _ in range(3)]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)

    # And the state is reachable, which is the only way to compare against
    # cv2 at more than one point.
    moved, _, _ = grab_cut(image, np.zeros((16, 20), np.uint8), rect=RECT,
                           iterations=2, mode="rect", rng_state=7)
    assert moved.shape == runs[0].shape


def test_grab_cut_is_the_same_whatever_the_plane_order():
    # A permutation of the colour axes permutes each component's mean and
    # conjugates its covariance, which leaves the Mahalanobis distance and
    # the determinant alone; the edge costs are sums of exactly representable
    # integers. So RGB and BGR give the same mask, which is what let the
    # caller drop its swap.
    image = scene()
    flipped = np.ascontiguousarray(image[:, :, ::-1])

    a, _, _ = grab_cut(image, np.zeros((16, 20), np.uint8), rect=RECT,
                       iterations=2, mode="rect")
    b, _, _ = grab_cut(flipped, np.zeros((16, 20), np.uint8), rect=RECT,
                       iterations=2, mode="rect")
    np.testing.assert_array_equal(a, b)


def test_grab_cut_refuses_what_it_cannot_segment():
    image = scene()

    with pytest.raises(ValueError):
        grab_cut(image[:, :, :1], np.zeros((16, 20), np.uint8), mode="mask")
    with pytest.raises(ValueError):
        grab_cut(image, np.zeros((16, 20), np.uint8), mode="sideways")
    with pytest.raises(ValueError):
        grab_cut(image, np.full((16, 20), 9, np.uint8), mode="mask")
    with pytest.raises(ValueError):
        grab_cut(image, np.zeros((4, 4), np.uint8), mode="mask")
    # A mask that leaves one side empty has nothing to fit a mixture to --
    # and "probably background" counts as background for that purpose, so an
    # all-probable-background mask is as empty as an all-certain one.
    with pytest.raises(ValueError):
        grab_cut(image, np.zeros((16, 20), np.uint8), mode="mask")
    with pytest.raises(ValueError):
        grab_cut(image, np.full((16, 20), 2, np.uint8), mode="mask")
