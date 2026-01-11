import copy
import numpy as np
import torch
import ubelt as ub
from collections import OrderedDict


def ensure_json_serializable(dict_, normalize_containers=False, verbose=0):
    """
    Attempt to convert common types (e.g. numpy) into something json complient

    Convert numpy and tuples into lists

    Args:
        normalize_containers (bool, default=False):
            if True, normalizes dict containers to be standard python
            structures.

    Example:
        >>> data = ub.ddict(lambda: int)
        >>> data['foo'] = ub.ddict(lambda: int)
        >>> data['bar'] = np.array([1, 2, 3])
        >>> data['foo']['a'] = 1
        >>> data['foo']['b'] = torch.FloatTensor([1, 2, 3])
        >>> result = ensure_json_serializable(data, normalize_containers=True)
        >>> assert type(result) is dict
    """
    dict_ = copy.deepcopy(dict_)

    def _norm_container(c):
        if isinstance(c, dict):
            # Cast to a normal dictionary
            if isinstance(c, OrderedDict):
                if type(c) is not OrderedDict:
                    c = OrderedDict(c)
            else:
                if type(c) is not dict:
                    c = dict(c)
        return c

    # inplace convert any ndarrays to lists
    def _walk_json(data, prefix=[]):
        items = None
        if isinstance(data, list):
            items = enumerate(data)
        elif isinstance(data, tuple):
            items = enumerate(data)
        elif isinstance(data, dict):
            items = data.items()
        else:
            raise TypeError(type(data))

        root = prefix
        level = {}
        for key, value in items:
            level[key] = value

        # yield a dict so the user can choose to not walk down a path
        yield root, level

        for key, value in level.items():
            if isinstance(value, (dict, list, tuple)):
                path = prefix + [key]
                for _ in _walk_json(value, prefix=path):
                    yield _

    def _convert(dict_, root, key, new_value):
        d = dict_
        for k in root:
            d = d[k]
        d[key] = new_value

    def _flatmap(func, data):
        if isinstance(data, list):
            return [_flatmap(func, item) for item in data]
        else:
            return func(data)

    to_convert = []
    for root, level in ub.ProgIter(_walk_json(dict_), desc='walk json',
                                   verbose=verbose):
        for key, value in level.items():
            if isinstance(value, tuple):
                # Convert tuples on the fly so they become mutable
                new_value = list(value)
                _convert(dict_, root, key, new_value)
            elif isinstance(value, np.ndarray):
                new_value = value.tolist()
                if 0:
                    if len(value.shape) == 1:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = list(map(int, new_value))
                        elif value.dtype.kind in {'f'}:
                            new_value = list(map(float, new_value))
                        elif value.dtype.kind in {'c'}:
                            new_value = list(map(complex, new_value))
                        else:
                            pass
                    else:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = _flatmap(int, new_value)
                        elif value.dtype.kind in {'f'}:
                            new_value = _flatmap(float, new_value)
                        elif value.dtype.kind in {'c'}:
                            new_value = _flatmap(complex, new_value)
                        else:
                            pass
                            # raise TypeError(value.dtype)
                to_convert.append((root, key, new_value))
            elif isinstance(value, torch.Tensor):
                new_value = value.data.cpu().numpy().tolist()
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.int16, np.int32, np.int64,
                                    np.uint16, np.uint32, np.uint64)):
                new_value = int(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.float32, np.float64)):
                new_value = float(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.complex64, np.complex128)):
                new_value = complex(value)
                to_convert.append((root, key, new_value))
            elif hasattr(value, '__json__'):
                new_value = value.__json__()
                to_convert.append((root, key, new_value))
            elif normalize_containers:
                if isinstance(value, dict):
                    new_value = _norm_container(value)
                    to_convert.append((root, key, new_value))

    for root, key, new_value in to_convert:
        _convert(dict_, root, key, new_value)

    if normalize_containers:
        # normalize the outer layer
        dict_ = _norm_container(dict_)
    return dict_
