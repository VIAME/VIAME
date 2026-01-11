# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


def roundrobin(iterables):
    """
    Round robin, iteration strategy

    In constrast to the recipie in itertools docs, the number of initial
    iterables does not need to be known, so it may be very large. This is
    useful if you only intend to extract a fixed number of items from the
    resulting iterable. Startup is instantainous.

    References:
        https://docs.python.org/3.8/library/itertools.html

    Args:
        iterables : an iterable of iterables

    Example:
        >>> list(roundrobin(['ABC', 'D', 'EF']))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    curr_alive = map(iter, iterables)
    while curr_alive:
        next_alive = []
        for gen in curr_alive:
            try:
                yield next(gen)
            except StopIteration:
                pass
            else:
                next_alive.append(gen)
        curr_alive = next_alive
