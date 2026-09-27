"""Exact matching stays bounded in memory and deterministic across blocks."""
import numpy as np
import pytest
from viame.image_processing import matching


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('k', [1, 2, 5])
def test_nearest_matches_reference_with_ties(binary, k):
    rng = np.random.default_rng(13)
    train = rng.integers(0, 8, (1100, 17), dtype=np.uint8)
    query = train[:7].copy()
    train[1024:1031] = query
    if binary:
        distance = np.unpackbits(query[:, None, :] ^ train[None, :, :], axis=2).sum(2)
    else:
        distance = ((query[:, None, :].astype(float) - train[None, :, :]) ** 2).sum(2)
    expected = np.argsort(distance, axis=1, kind='stable')[:, :k]
    indices, found = matching.nearest(query, train, k, binary, with_distance=True)
    assert np.array_equal(indices, expected)
    expected_distance = np.take_along_axis(distance, expected, axis=1)
    assert np.allclose(found, expected_distance if binary else np.sqrt(expected_distance))


@pytest.mark.parametrize('binary', [False, True])
def test_ratio_rejects_duplicate_nearest_descriptors(binary):
    query = np.zeros((1, 32), np.uint8)
    assert matching.ratio_match(query, np.zeros((1100, 32), np.uint8), binary=binary) == []


def test_binary_search_does_not_unpack_descriptors(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('bit expansion must not be used by the native search')
    monkeypatch.setattr(np, 'unpackbits', forbidden)
    data = np.random.default_rng(6).integers(0, 256, (1000, 32), dtype=np.uint8)
    assert np.array_equal(matching.nearest(data, data, 1, True)[:, 0], np.arange(1000))
