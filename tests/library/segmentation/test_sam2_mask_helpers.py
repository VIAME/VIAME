"""SAM2 mask cleanup and point selection without an OpenCV runtime."""
import numpy as np
import pytest


def test_remove_small_regions():
    from viame.sam2.utils.amg import remove_small_regions
    mask = np.zeros((12,13),dtype=bool)
    mask[2:9,3:10] = True
    mask[4,5] = False
    mask[0,0] = True
    holes,changed = remove_small_regions(mask,3,'holes')
    assert changed and holes[4,5] and holes[0,0]
    islands,changed = remove_small_regions(holes,3,'islands')
    assert changed and not islands[0,0]
    assert islands.sum() == 49


@pytest.mark.parametrize('padding', [False,True])
def test_center_point_is_inside_false_negative(padding):
    import torch
    from viame.sam2.modeling.sam2_utils import sample_one_point_from_error_center
    target = torch.zeros((1,1,13,15),dtype=torch.bool)
    target[:,:,3:10,4:11] = True
    point,label = sample_one_point_from_error_center(target,None,padding=padding)
    assert point.tolist() == [[[7.,6.]]]
    assert label.tolist() == [[1]]


def test_center_point_without_boundary_chooses_first_pixel():
    import torch
    from viame.sam2.modeling.sam2_utils import sample_one_point_from_error_center
    target = torch.ones((1,1,7,9),dtype=torch.bool)
    point,label = sample_one_point_from_error_center(target,None,padding=False)
    assert point.tolist() == [[[0.,0.]]]
    assert label.tolist() == [[1]]


def test_equal_small_islands_preserve_opencv_block_order():
    from viame.sam2.utils.amg import remove_small_regions
    mask = np.zeros((7, 9), dtype=bool)
    mask[0, 4] = True
    mask[1, 0] = True
    result, changed = remove_small_regions(mask, 2, 'islands')
    assert changed
    assert result.sum() == 1
    assert result[1, 0]
