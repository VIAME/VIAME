"""The 3D layout the query service computes for result visualization."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, '..', '..', '..', 'plugins', 'core', 'query_service.py')


@pytest.fixture(scope='module')
def embed_around():
    spec = importlib.util.spec_from_file_location('query_service_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.embed_around


def test_query_sits_at_origin_and_radii_follow_descriptor_distances(embed_around):
    rng = np.random.default_rng(0)
    center = rng.normal(size=16)
    # Three points at known distances along orthogonal directions.
    basis = np.linalg.qr(rng.normal(size=(16, 3)))[0].T
    vectors = center + basis * np.array([[1.0], [2.0], [3.0]])
    positions, distances = embed_around(center, vectors)
    assert distances == pytest.approx([1.0, 2.0, 3.0])
    assert np.linalg.norm(positions, axis=1) == pytest.approx([1.0, 2.0, 3.0])
    assert np.linalg.norm(np.array(positions[0]) - positions[1]) == pytest.approx(np.sqrt(5.0))


def test_layout_is_deterministic_and_pads_to_three_axes(embed_around):
    center = np.zeros(4)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    first, _ = embed_around(center, vectors)
    second, _ = embed_around(center, vectors)
    assert first == second
    assert all(len(p) == 3 for p in first)
    assert [p[2] for p in first] == [0.0, 0.0]
    assert embed_around(center, np.zeros((0, 4))) == ([], [])
