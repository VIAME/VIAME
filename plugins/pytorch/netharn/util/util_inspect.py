import six


def default_kwargs(cls):
    """
    Grab initkw defaults from the constructor

    Args:
        cls (type | callable): a class or function

    Example:
        >>> from .util.util_inspect import *  # NOQA
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> import torch
        >>> import ubelt as ub
        >>> cls = torch.optim.Adam
        >>> default_kwargs(cls)
        >>> cls = nh.initializers.KaimingNormal
        >>> print(ub.repr2(default_kwargs(cls), nl=0))
        {'mode': 'fan_in', 'param': 0}
        >>> cls = nh.initializers.NoOp
        >>> default_kwargs(cls)
        {}

    SeeAlso:
        xinspect.get_func_kwargs(cls)
    """
    if six.PY2:
        if cls.__init__ is object.__init__:
            # hack for python2 classes without __init__
            return {}
        else:
            import funcsigs
            sig = funcsigs.signature(cls)
    else:
        import inspect
        sig = inspect.signature(cls)

    default_kwargs = {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not p.empty
    }
    return default_kwargs
