"""
SMQTK Bit Utilities - Bit vector manipulation functions.
"""
import numpy

from . import ncr


def bit_vector_to_int_large(v):
    """
    Transform a numpy vector representing a sequence of binary bits [0 | >0]
    into an integer representation.

    This function handles very large integers (>64bit).
    """
    c = 0
    for b in v:
        c = (c << 1) + int(b)
    return c


def int_to_bit_vector_large(integer, bits=0):
    """
    Transform integer into a bit vector, optionally of a specific length.

    This function handles very large integers (>64bit).
    """
    size = len(bin(integer)) - 2

    if bits and (bits - size) < 0:
        raise ValueError("%d bits too small to represent integer value %d."
                         % (bits, integer))

    v = numpy.zeros(bits or size, numpy.bool_)
    for i in range(0, size):
        v[-(i+1)] = integer & 1
        integer >>= 1

    return v


def next_perm(v):
    """
    Compute the lexicographically next bit permutation.
    """
    t = (v | (v - 1)) + 1
    w = t | ((((t & -t) // (v & -v)) >> 1) - 1)
    return w


def iter_perms(l, n):
    """
    Return an iterator over bit combinations of length l with n set bits.
    """
    if n <= 0:
        return
    n = min(l, n)
    s = (1 << n) - 1
    yield s
    for _ in range(ncr(l, n) - 1):
        s = next_perm(s)
        yield s


def neighbor_codes(b, c, d):
    """
    Iterate through integers of bit length b that are d hamming distance
    away from query code c.
    """
    if not d:
        yield c
    else:
        for fltr in iter_perms(b, d):
            yield c ^ fltr


__all__ = [
    'bit_vector_to_int_large',
    'int_to_bit_vector_large',
    'next_perm',
    'iter_perms',
    'neighbor_codes',
]
