# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for the Interactive Stereo Service.

Tests cover:
1. Module import and class instantiation
2. InteractiveStereoService class methods
3. JSON protocol handling (mocked algorithm)
4. Calibration loading
5. Disparity cache management
6. Line/point transfer calculations

These tests use mocked algorithms to avoid requiring actual stereo models.
"""

import json
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pytest
from PIL import Image


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_image(path, width=640, height=480):
    """Create a test image with some synthetic content."""
    im = Image.effect_mandelbrot(
        (width, height),
        (-2.23845, -1.1538375, 0.83845, 1.1538375),
        64
    )
    im = im.convert('RGB')
    im.save(path)
    return path


def create_stereo_image_pair(dir_path, width=640, height=480):
    """Create a pair of stereo test images."""
    left_path = os.path.join(dir_path, 'left.png')
    right_path = os.path.join(dir_path, 'right.png')

    # Create slightly different images for left/right
    left_im = Image.effect_mandelbrot(
        (width, height),
        (-2.23845, -1.1538375, 0.83845, 1.1538375),
        64
    )
    right_im = Image.effect_mandelbrot(
        (width, height),
        (-2.13845, -1.0538375, 0.93845, 1.2538375),
        64
    )

    left_im.convert('RGB').save(left_path)
    right_im.convert('RGB').save(right_path)

    return left_path, right_path


class MockStereoAlgorithm:
    """Mock ComputeStereoDepthMap algorithm for testing."""

    def __init__(self, disparity_value=50.0):
        self.disparity_value = disparity_value
        self.compute_called = False
        self.last_left_image = None
        self.last_right_image = None

    def compute(self, left_image, right_image):
        """Return a mock disparity map."""
        self.compute_called = True
        self.last_left_image = left_image
        self.last_right_image = right_image

        # Get image dimensions from left image
        img = left_image.image()
        img_array = img.asarray()
        height, width = img_array.shape[:2]

        # Create a uniform disparity map (scaled by 256 as uint16)
        disparity = np.full((height, width), self.disparity_value * 256, dtype=np.uint16)

        # Return as ImageContainer
        from kwiver.vital.types import ImageContainer, Image
        return ImageContainer(Image(disparity))


class MockImageContainer:
    """Mock ImageContainer for testing without KWIVER."""

    def __init__(self, array):
        self._array = array

    def image(self):
        return MockImage(self._array)


class MockImage:
    """Mock Image for testing without KWIVER."""

    def __init__(self, array):
        self._array = array

    def asarray(self):
        return self._array


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_test_dir():
    """Fixture providing a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def stereo_images(temp_test_dir):
    """Fixture providing a pair of stereo test images."""
    return create_stereo_image_pair(temp_test_dir)


@pytest.fixture
def mock_stereo_algo():
    """Fixture providing a mock stereo algorithm."""
    return MockStereoAlgorithm(disparity_value=50.0)


# =============================================================================
# Module Import Tests
# =============================================================================

class TestInteractiveStereoImports:
    """Tests for module imports."""

    def test_import_interactive_stereo(self):
        """Test that interactive_stereo module can be imported."""
        from viame.core import interactive_stereo
        assert hasattr(interactive_stereo, 'InteractiveStereoService')
        assert hasattr(interactive_stereo, 'load_algorithm_from_config')
        assert hasattr(interactive_stereo, 'create_default_config')

    def test_interactive_stereo_service_class(self):
        """Test that InteractiveStereoService class has expected methods."""
        from viame.core.interactive_stereo import InteractiveStereoService

        # Check key methods exist
        assert hasattr(InteractiveStereoService, 'handle_enable')
        assert hasattr(InteractiveStereoService, 'handle_disable')
        assert hasattr(InteractiveStereoService, 'handle_set_frame')
        assert hasattr(InteractiveStereoService, 'handle_transfer_line')
        assert hasattr(InteractiveStereoService, 'handle_transfer_points')
        assert hasattr(InteractiveStereoService, 'handle_get_status')
        assert hasattr(InteractiveStereoService, 'handle_cancel')
        assert hasattr(InteractiveStereoService, 'run')


# =============================================================================
# InteractiveStereoService Unit Tests
# =============================================================================

class TestInteractiveStereoService:
    """Unit tests for InteractiveStereoService class."""

    def test_service_initialization(self, mock_stereo_algo):
        """Test service initialization with mock algorithm."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
            scale=1.0
        )

        assert service._stereo_algo is mock_stereo_algo
        assert service._scale == 1.0
        assert service._enabled is False
        assert service._disparity_ready is False
        assert service._current_disparity is None

    def test_load_calibration(self, mock_stereo_algo):
        """Test calibration loading."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        calibration = {
            'fx_left': 1000.0,
            'fy_left': 1000.0,
            'cx_left': 320.0,
            'cy_left': 240.0,
            'T': [-0.1, 0.0, 0.0],  # 10cm baseline
        }

        service._load_calibration(calibration)

        assert service._focal_length == 1000.0
        assert service._baseline == 0.1
        assert service._principal_x == 320.0
        assert service._principal_y == 240.0
        assert service._calibration == calibration

    def test_load_calibration_magnitude_baseline(self, mock_stereo_algo):
        """Test calibration with non-horizontal baseline."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        # Translation vector not purely horizontal
        calibration = {
            'fx_left': 1000.0,
            'T': [0.0, 0.0, 0.1],  # 10cm in Z direction
        }

        service._load_calibration(calibration)

        # Should use magnitude since X is near zero
        assert abs(service._baseline - 0.1) < 0.001

    def test_handle_enable(self, mock_stereo_algo):
        """Test enable handler."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        calibration = {
            'fx_left': 1000.0,
            'T': [-0.1, 0.0, 0.0],
        }

        result = service.handle_enable({'calibration': calibration})

        assert result['success'] is True
        assert service._enabled is True
        assert service._focal_length == 1000.0

    def test_handle_enable_already_enabled(self, mock_stereo_algo):
        """Test enable when already enabled."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        result = service.handle_enable({})

        assert result['success'] is True
        assert 'Already enabled' in result['message']

    def test_handle_disable(self, mock_stereo_algo):
        """Test disable handler."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        result = service.handle_disable({})

        assert result['success'] is True
        assert service._enabled is False

    def test_handle_get_status(self, mock_stereo_algo):
        """Test status handler."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True
        service._disparity_ready = True
        service._current_left_path = '/path/to/left.png'
        service._current_right_path = '/path/to/right.png'

        result = service.handle_get_status({})

        assert result['success'] is True
        assert result['enabled'] is True
        assert result['disparity_ready'] is True
        assert result['current_left_path'] == '/path/to/left.png'

    def test_handle_set_calibration(self, mock_stereo_algo):
        """Test set_calibration handler."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        calibration = {
            'fx_left': 2000.0,
            'T': [-0.2, 0.0, 0.0],
        }

        result = service.handle_set_calibration({'calibration': calibration})

        assert result['success'] is True
        assert service._focal_length == 2000.0
        assert service._baseline == 0.2

    def test_handle_set_calibration_missing(self, mock_stereo_algo):
        """Test set_calibration without calibration data."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        with pytest.raises(ValueError, match="calibration is required"):
            service.handle_set_calibration({})


# =============================================================================
# Disparity Cache Tests
# =============================================================================

class TestDisparityCache:
    """Tests for disparity caching functionality."""

    def test_add_to_cache(self, mock_stereo_algo):
        """Test adding disparity to cache."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        disparity = np.ones((480, 640), dtype=np.float32) * 50.0

        service._add_to_cache('/left1.png', '/right1.png', disparity)

        assert len(service._disparity_cache) == 1
        assert len(service._cache_order) == 1
        assert ('/left1.png', '/right1.png') in service._disparity_cache

    def test_get_from_cache(self, mock_stereo_algo):
        """Test retrieving disparity from cache."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        disparity = np.ones((480, 640), dtype=np.float32) * 50.0
        service._add_to_cache('/left1.png', '/right1.png', disparity)

        result = service._get_from_cache('/left1.png', '/right1.png')

        assert result is not None
        np.testing.assert_array_equal(result, disparity)

    def test_cache_miss(self, mock_stereo_algo):
        """Test cache miss returns None."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        result = service._get_from_cache('/nonexistent.png', '/other.png')

        assert result is None

    def test_cache_eviction(self, mock_stereo_algo):
        """Test LRU cache eviction."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._max_cache_size = 3

        # Add 4 items (should evict first)
        for i in range(4):
            disparity = np.ones((480, 640), dtype=np.float32) * (i + 1)
            service._add_to_cache(f'/left{i}.png', f'/right{i}.png', disparity)

        assert len(service._disparity_cache) == 3
        # First item should be evicted
        assert ('/left0.png', '/right0.png') not in service._disparity_cache
        # Last 3 should remain
        assert ('/left1.png', '/right1.png') in service._disparity_cache
        assert ('/left2.png', '/right2.png') in service._disparity_cache
        assert ('/left3.png', '/right3.png') in service._disparity_cache

    def test_cache_lru_update(self, mock_stereo_algo):
        """Test LRU order is updated on cache hit."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._max_cache_size = 3

        # Add 3 items
        for i in range(3):
            disparity = np.ones((480, 640), dtype=np.float32) * (i + 1)
            service._add_to_cache(f'/left{i}.png', f'/right{i}.png', disparity)

        # Access first item (moves it to end of LRU list)
        service._get_from_cache('/left0.png', '/right0.png')

        # Add new item (should evict left1, not left0)
        disparity = np.ones((480, 640), dtype=np.float32) * 99
        service._add_to_cache('/left99.png', '/right99.png', disparity)

        assert ('/left0.png', '/right0.png') in service._disparity_cache
        assert ('/left1.png', '/right1.png') not in service._disparity_cache


# =============================================================================
# Line/Point Transfer Tests
# =============================================================================

class TestLinePointTransfer:
    """Tests for line and point transfer using disparity."""

    def test_transfer_line_basic(self, mock_stereo_algo):
        """Test basic line transfer."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        # Set up a uniform disparity map (50 pixels)
        disparity = np.ones((480, 640), dtype=np.float32) * 50.0
        service._current_disparity = disparity
        service._disparity_ready = True

        request = {
            'line': [[100, 200], [300, 200]]
        }

        result = service.handle_transfer_line(request)

        assert result['success'] is True
        transferred = result['transferred_line']
        # x_right = x_left - disparity (50)
        assert transferred[0][0] == 50.0  # 100 - 50
        assert transferred[0][1] == 200.0  # y unchanged
        assert transferred[1][0] == 250.0  # 300 - 50
        assert transferred[1][1] == 200.0  # y unchanged

    def test_transfer_line_with_depth(self, mock_stereo_algo):
        """Test line transfer with depth computation."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True
        service._focal_length = 1000.0
        service._baseline = 0.1  # 10cm

        # Set up a uniform disparity map
        disparity = np.ones((480, 640), dtype=np.float32) * 50.0
        service._current_disparity = disparity
        service._disparity_ready = True

        request = {
            'line': [[100, 200], [300, 200]]
        }

        result = service.handle_transfer_line(request)

        assert result['success'] is True
        assert result['depth_info'] is not None
        # depth = focal * baseline / disparity = 1000 * 0.1 / 50 = 2.0
        assert abs(result['depth_info']['depth_point1'] - 2.0) < 0.01
        assert abs(result['depth_info']['depth_point2'] - 2.0) < 0.01

    def test_transfer_line_not_enabled(self, mock_stereo_algo):
        """Test line transfer when service not enabled."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        with pytest.raises(ValueError, match="Service not enabled"):
            service.handle_transfer_line({'line': [[0, 0], [100, 100]]})

    def test_transfer_line_disparity_not_ready(self, mock_stereo_algo):
        """Test line transfer when disparity not ready."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True
        service._disparity_ready = False

        with pytest.raises(ValueError, match="Disparity not ready"):
            service.handle_transfer_line({'line': [[0, 0], [100, 100]]})

    def test_transfer_points_basic(self, mock_stereo_algo):
        """Test basic point transfer."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        # Set up a uniform disparity map (50 pixels)
        disparity = np.ones((480, 640), dtype=np.float32) * 50.0
        service._current_disparity = disparity
        service._disparity_ready = True

        request = {
            'points': [[100, 200], [300, 200], [200, 300]]
        }

        result = service.handle_transfer_points(request)

        assert result['success'] is True
        transferred = result['transferred_points']
        assert len(transferred) == 3
        assert transferred[0] == [50.0, 200.0]
        assert transferred[1] == [250.0, 200.0]
        assert transferred[2] == [150.0, 300.0]

    def test_transfer_points_varying_disparity(self, mock_stereo_algo):
        """Test point transfer with varying disparity values."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        # Create disparity map with varying values
        disparity = np.zeros((480, 640), dtype=np.float32)
        disparity[200, 100] = 30.0
        disparity[200, 300] = 60.0
        service._current_disparity = disparity
        service._disparity_ready = True

        request = {
            'points': [[100, 200], [300, 200]]
        }

        result = service.handle_transfer_points(request)

        assert result['success'] is True
        assert result['disparity_values'] == [30.0, 60.0]
        assert result['transferred_points'][0] == [70.0, 200.0]  # 100 - 30
        assert result['transferred_points'][1] == [240.0, 200.0]  # 300 - 60


# =============================================================================
# Request Handler Tests
# =============================================================================

class TestRequestHandler:
    """Tests for request routing."""

    def test_handle_request_routing(self, mock_stereo_algo):
        """Test that handle_request routes to correct handlers."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        # Test enable routing
        result = service.handle_request({'command': 'enable'})
        assert result['success'] is True

        # Test get_status routing
        result = service.handle_request({'command': 'get_status'})
        assert 'enabled' in result

        # Test disable routing
        result = service.handle_request({'command': 'disable'})
        assert result['success'] is True

    def test_handle_request_unknown_command(self, mock_stereo_algo):
        """Test unknown command raises error."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        with pytest.raises(ValueError, match="Unknown command"):
            service.handle_request({'command': 'invalid_command'})


# =============================================================================
# Config File Tests
# =============================================================================

class TestConfigFile:
    """Tests for config file generation and parsing."""

    def test_create_default_config(self, temp_test_dir):
        """Test default config file creation."""
        from viame.core.interactive_stereo import create_default_config

        config_path = os.path.join(temp_test_dir, 'test_config.conf')
        create_default_config(config_path)

        assert os.path.exists(config_path)

        with open(config_path, 'r') as f:
            content = f.read()

        assert 'matching_method = epipolar_template_matching' in content
        assert 'service:scale' in content


# =============================================================================
# Integration Tests (with mocked image loading)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked dependencies."""

    def test_set_frame_cached(self, mock_stereo_algo, stereo_images):
        """Test set_frame with cached disparity."""
        from viame.core.interactive_stereo import InteractiveStereoService

        left_path, right_path = stereo_images

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        # Pre-populate cache
        disparity = np.ones((480, 640), dtype=np.float32) * 50.0
        service._add_to_cache(left_path, right_path, disparity)

        result = service.handle_set_frame({
            'left_image_path': left_path,
            'right_image_path': right_path,
        })

        assert result['success'] is True
        assert result['disparity_ready'] is True
        assert 'cache' in result['message'].lower()

    def test_set_frame_already_ready(self, mock_stereo_algo, stereo_images):
        """Test set_frame when disparity already computed for current frame."""
        from viame.core.interactive_stereo import InteractiveStereoService

        left_path, right_path = stereo_images

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True
        service._current_left_path = left_path
        service._current_right_path = right_path
        service._disparity_ready = True

        result = service.handle_set_frame({
            'left_image_path': left_path,
            'right_image_path': right_path,
        })

        assert result['success'] is True
        assert result['disparity_ready'] is True

    def test_set_frame_not_enabled(self, mock_stereo_algo, stereo_images):
        """Test set_frame when service not enabled."""
        from viame.core.interactive_stereo import InteractiveStereoService

        left_path, right_path = stereo_images

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )

        with pytest.raises(ValueError, match="Service not enabled"):
            service.handle_set_frame({
                'left_image_path': left_path,
                'right_image_path': right_path,
            })

    def test_set_frame_missing_paths(self, mock_stereo_algo):
        """Test set_frame with missing image paths."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        with pytest.raises(ValueError, match="left_image_path and right_image_path are required"):
            service.handle_set_frame({})

    def test_set_frame_nonexistent_images(self, mock_stereo_algo):
        """Test set_frame with nonexistent images."""
        from viame.core.interactive_stereo import InteractiveStereoService

        service = InteractiveStereoService(
            compute_stereo_depth_map_algo=mock_stereo_algo,
        )
        service._enabled = True

        with pytest.raises(ValueError, match="not found"):
            service.handle_set_frame({
                'left_image_path': '/nonexistent/left.png',
                'right_image_path': '/nonexistent/right.png',
            })


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
