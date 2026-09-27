"""Feature array bindings must allow other Python threads to make progress."""
import threading
import numpy as np
import pytest
from viame.image_processing import features


@pytest.mark.parametrize('operation', ['sift', 'surf', 'sift_describe'])
def test_native_feature_work_releases_gil(operation):
    image = np.random.default_rng(37).integers(0, 256, (768, 768), dtype=np.uint8)
    keypoints = None
    if operation == 'sift_describe':
        keypoints, _ = features.sift(image, n_features=2000, describe=False)
    progressed = threading.Event()
    # The timer cannot execute Python while the detector holds the GIL.
    timer = threading.Timer(0.005, progressed.set)
    timer.start()
    try:
        if keypoints is None:
            found, described = getattr(features, operation)(image)
        else:
            found, described = features.sift_describe(image, keypoints)
        assert progressed.is_set(), 'feature computation held the GIL'
        assert len(found) == len(described)
    finally:
        timer.cancel()
        timer.join()
