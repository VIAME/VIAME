# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import ubelt as ub


def compact_idstr(dict_):
    """
    A short unique id string for a dict param config that is semi-interpretable
    """
    from netharn import util
    import ubelt as ub
    short_keys = util.shortest_unique_prefixes(dict_.keys())
    short_dict = ub.odict(sorted(zip(short_keys, dict_.values())))
    idstr = ub.repr2(short_dict, nobr=1, itemsep='', si=1, nl=0,
                     explicit=1)
    return idstr


def make_idstr(d):
    """
    Make full-length-key id-string
    """
    if d is None:
        return ''
    elif isinstance(d, six.string_types):
        return d
    elif len(d) == 0:
        return ''
    if not isinstance(d, ub.odict):
        d = ub.odict(sorted(d.items()))
    return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0, si=True)


def make_short_idstr(params, precision=None):
    """
    Make id-string where they keys are shortened

    Args:
        params (dict): a configuration dictionary to be summarized
        precision (int): maximum number of decimal points for float values

    Returns:
        str: a short string roughly summarizing the dictionary contents

    CommandLine:
        python -m netharn.util.misc make_short_idstr

    Example:
        >>> # xdoctest: +SKIP
        >>> from .util.util_idstr import *  # NOQA
        >>> params = {'input_shape': (None, 3, 212, 212),
        >>>           'a': 'b',
        >>>           'center': {'im_mean': .5, 'std': 1},
        >>>           'alphabet': 'abc'}
        >>> print(make_short_idstr(params))
        a=b,al=abc,c=dictim_mean=0.5,std=1,i=None,3,212,212
    """
    if params is None:
        return ''
    elif len(params) == 0:
        return ''
    from netharn import util
    short_keys = util.shortest_unique_prefixes(list(params.keys()),
                                               allow_simple=False,
                                               allow_end=True,
                                               min_length=1)
    def shortval(v):
        if isinstance(v, bool):
            return int(v)
        return v
    d = dict(zip(short_keys, map(shortval, params.values())))
    def make_idstr(d):
        # Note: we are not using sort=True, because repr2 sorts sets and dicts
        # by default.
        remove_chars = [' ', '[', ']', '(', ')', '{', '}']
        idstr = ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0, si=True,
                         precision=precision)
        for c in remove_chars:
            idstr = idstr.replace(c, '')
        return idstr

    short_idstr = make_idstr(d)
    return short_idstr
