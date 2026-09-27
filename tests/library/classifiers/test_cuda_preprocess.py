"""Optional classifier input parity; no downloaded models needed."""
import numpy as np
import pytest


def test_classifier_batches_match_cpu():
    torch = pytest.importorskip("torch")
    kwimage = pytest.importorskip("kwimage")
    from viame.image_kernels import cuda
    from viame.classifiers.cuda_preprocess import CUDAClassifierBatches
    if not cuda.available() or not torch.cuda.is_available():
        pytest.skip("native/PyTorch CUDA unavailable")
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
              for h, w in [(64, 75), (800, 973), (224, 224), (112, 224), (316, 523)]]
    batches = CUDAClassifierBatches(images, (224, 224), 2, "cuda:0")
    assert len(batches) == 3
    actual = torch.cat([b["inputs"]["rgb"] for b in batches]).cpu().numpy()
    expected = np.stack([(kwimage.imresize(x, dsize=(224, 224), letterbox=True)
                          .transpose(2, 0, 1) / 255.).astype(np.float32) for x in images])
    np.testing.assert_array_equal(actual, expected)
    assert list(CUDAClassifierBatches([], (224, 224), 2, "cuda:0")) == []
    with pytest.raises(ValueError):
        CUDAClassifierBatches(images, (224, 224), 2, "cpu")


def test_gfit_checkpoint_predictions():
    """Opt-in model test: VIAME_GFIT_CLASSIFIER_MODEL=/path/to/deployed.zip."""
    import os
    path = os.environ.get("VIAME_GFIT_CLASSIFIER_MODEL")
    if not path:
        pytest.skip("VIAME_GFIT_CLASSIFIER_MODEL not set")
    from viame.object_detectors.netharn.netharn.clf_predict import ClfPredictConfig, ClfPredictor
    cfg = ClfPredictConfig()
    cfg["deployed"] = path
    cfg["batch_size"] = 2
    cfg["xpu"] = "0"
    cfg["verbose"] = 0
    predictor = ClfPredictor(cfg)
    rng = np.random.default_rng(78)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
              for h, w in [(61, 75), (411, 519), (224, 224), (76, 300)]]
    expected = list(predictor.predict(images))
    predictor.config["preprocess_backend"] = "cuda"
    actual = list(predictor.predict(images))
    assert len(expected) == len(actual) == len(images)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left.data["prob"], right.data["prob"])
