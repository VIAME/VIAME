"""
Implementation of
Layer rotation: a surprisingly powerful indicator of generalization in deep
networks?

References:
    https://arxiv.org/pdf/1806.01603.pdf
    https://github.com/vfdev-5/LayerRotation-pytorch/blob/master/code/handlers/layer_rotation.py
"""
import ubelt as ub
import numpy as np
import torch


def _get_named_params(model, copy=False):
    """
    Example:
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> dict(_get_named_params(model)).keys()
    """
    def fn(p):
        p = p.cpu().detach()
        if copy:
            p = p.clone()
        return p
    named_params = [
        (key, fn(p)) for key, p in model.named_parameters()
        if 'weight' in key
    ]
    return named_params


def layer_rotation(current_params, init_params):
    """
    Example:
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> model2 = nh.models.ToyNet2d()
        >>> init_params = _get_named_params(model)
        >>> current_params = _get_named_params(model2)
        >>> ret = layer_rotation(current_params, init_params)

    """
    ret = []
    for (n1, p1), (n2, p2) in zip(current_params, init_params):
        assert n1 == n2, "{} vs {}".format(n1, n2)
        sim = torch.cosine_similarity(p1.reshape(-1), p2.reshape(-1), dim=0).item()
        dist = 1.0 - sim
        ret.append((n1, dist))
    return ret


class LayerRotation(ub.NiceRepr):
    """
    Example:
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> self = LayerRotation(model)
        >>> self.measure()
        >>> nh.initializers.KaimingNormal()(self.model)
        >>> self.measure()
    """
    def __init__(self, model):
        self.model = model
        self.init_params = _get_named_params(model, copy=True)
        self.stats = None

    def __nice__(self):
        return ub.repr2(self.stats)

    def measure(self):
        import kwarray
        current_params = _get_named_params(self.model)
        ret = layer_rotation(current_params, self.init_params)
        values = np.array([v for n, v in ret])
        self.stats = kwarray.stats_dict(values, median=True)
        return self.stats
